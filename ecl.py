# -*- coding: utf-8 -*-
# ecl.py
# -------------------------------------------------------------------
# IFRS 9 Expected Credit Loss Engine (12-month & Lifetime)
# - Inputs:
#     artifacts/snapshot_scored.parquet   (from modeling.py)
#     artifacts/ifrs9_panel.parquet       (optional; for staging backstops)
# - Features:
#     • Stage 1/2/3 belirleme (SICR, 30/60/90dpd backstop, default)
#     • 12A & Lifetime ECL (aylık hazard agregasyonu)
#     • Senaryo seti: baseline / stress / recovery (ağırlıklı)
#     • EAD path: amortizasyon + prepayment heterojenliği + revolving draw
#     • LGD path: downturn uplift + stokastik time-to-sale (beklenen değerle)
#     • Stage 3 accrual stop: faizi post-default sayma
# - Outputs (under artifacts/ecl/):
#     account_ecl.parquet  (hesap bazlı)
#     summary_by_segment.csv / summary_by_stage.csv / summary_by_scenario.csv
#     portfolio_summary.json
# -------------------------------------------------------------------

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------
# IO & Constants
# ---------------------------
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)
ECL_DIR = ART / "ecl"
ECL_DIR.mkdir(parents=True, exist_ok=True)

SNAP_PATH   = ART / "snapshot_scored.parquet"
PANEL_PATH  = ART / "ifrs9_panel.parquet"  # optional

MONTHS_HORIZON = int(os.getenv("IFRS9_HORIZON_MONTHS", "36"))
DISCOUNT_SPREAD = float(os.getenv("IFRS9_DISCOUNT_SPREAD", "0.03"))

RECOVERY_LAG_MIN, RECOVERY_LAG_MAX = 6, 24    # months (uniform EV ~ 15)
PREPAY_BASE = 0.004                           # base monthly prepayment
AMORT_RATE_INST = 0.015                       # installment amortization baseline
AMORT_RATE_REV  = 0.006                       # revolving amortization baseline (slower)
DRAW_ON_DEFAULT_REV = 0.10                    # undrawn drawdown on default
CCF_NON_REV  = 0.02                           # small add for non-rev scaling

SCENARIOS = {
    # macro deltalarda hazard & lgd tepkisini aşağıda hesaplayacağız
    "baseline": {"weight": 0.60, "unemp": 0.10, "gdp": 0.02,  "policy": 0.30},
    "stress":   {"weight": 0.30, "unemp": 0.18, "gdp": -0.03, "policy": 0.40},
    "recovery": {"weight": 0.10, "unemp": 0.08, "gdp": 0.04,  "policy": 0.28},
}

REV_NAMES = {"CreditCard","Overdraft","Revolving","revolving"}

# ---------------------------
# Helpers
# ---------------------------
def load_inputs(snap_path=SNAP_PATH, panel_path=PANEL_PATH):
    if not Path(snap_path).exists():
        raise FileNotFoundError(f"Snapshot not found: {snap_path}. Run modeling.py first.")
    snap = pd.read_parquet(snap_path)
    panel = pd.read_parquet(panel_path) if Path(panel_path).exists() else None
    return snap, panel

def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

def monthly_discount_factors(rate_annual, months):
    r_m = np.clip(rate_annual, 0.0, 2.0) / 12.0
    t = np.arange(1, months+1)
    return (1.0 / np.power(1.0 + r_m, t)).astype(np.float64)

def hazard_from_pd12(pd12):
    pd12 = np.clip(pd12, 1e-9, 0.999999)
    h = 1 - np.power(1 - pd12, 1/12.0)
    return np.clip(h, 1e-6, 0.999)

def default_month_pmf(mhaz):
    n, T = mhaz.shape
    surv = np.cumprod(1 - mhaz, axis=1)
    surv_shift = np.hstack([np.ones((n,1)), surv[:,:-1]])
    pmf = surv_shift * mhaz
    return pmf

def scenario_adjust_hazard(pd12_base, unemp, gdp):
    """
    Macro duyarlılık: işsizlik ↑ → hazard ↑ ; büyüme ↓ → hazard ↑
    Lineer bir çarpan uygulanır (realistic proxy).
    """
    # referans ~ 10% işsizlik, 2% büyüme
    du = unemp - 0.10
    dg = 0.02 - gdp
    mult = 1.0 + 1.6*max(0.0, du) + 0.9*max(0.0, dg)
    h = hazard_from_pd12(pd12_base) * mult
    return np.clip(h, 1e-6, 0.999)

def seasoning_shape(months, seasoning=None):
    """
    Weibull-vari hump: erken dönem risk yüksek, sonra düşer.
    Seasoning bilgisi yoksa tek biçimli şekil uygula.
    """
    t = np.arange(1, months+1)
    shape = (0.9*np.exp(-((t-10)/10.0)**2) + 0.2/(1+0.02*t))
    if seasoning is None:
        return shape
    # seasoning ile erken/yüksek seasoning farkını ufak etkilerle yansıt
    s = np.clip(seasoning.astype(float), 0, 120)
    adj = (1 - 0.002*s)  # seasoning arttıkça hazard şekli biraz azalır
    return shape * adj[:,None]

def ead_path(ead0, months, is_revolving, risk_score=None):
    """
    EAD(t): amortizasyon + prepayment heterojenliği (risk_score yüksek → daha çok prepay)
    """
    ead0 = ead0.astype(float)
    rs = np.zeros_like(ead0) + (50.0 if risk_score is None else np.clip(risk_score, 0, 100))
    prepay = PREPAY_BASE * (0.6 + 0.008*rs)  # 0.6..1.4 × base
    amort = np.where(is_revolving, AMORT_RATE_REV, AMORT_RATE_INST)
    decay = amort + prepay
    T = np.arange(0, months)
    return ead0[:,None] * np.exp(-decay[:,None]*T[None,:])

def lgd_curve(lgd0, months, unemp, gdp):
    """
    Downturn uplift + time-to-sale (EV) ile LGD(t) ~ sabit seviye + makro etkisi
    """
    downturn = 0.10*max(0.0, unemp-0.10) + 0.06*max(0.0, -gdp)
    ev_lag = 0.5*(RECOVERY_LAG_MIN + RECOVERY_LAG_MAX)  # beklenen lag
    lag_uplift = 0.10*(ev_lag/24.0)                     # 0..0.1 ~
    level = np.clip(lgd0 + downturn + lag_uplift, 0.01, 0.99)
    return np.repeat(level[:,None], months, axis=1)

def revolving_draw_expectation(ead_path_t, undrawn, is_revolving, pmf, ccf=None):
    # If ccf is provided per-account, use it; else fallback to global
    if ccf is None:
        ccf_mat = DRAW_ON_DEFAULT_REV * np.ones_like(ead_path_t)
    else:
        ccf_mat = np.repeat(np.clip(ccf, 0.0, 1.0)[:,None], ead_path_t.shape[1], axis=1)
    add = ccf_mat * undrawn[:,None] * is_revolving[:,None]
    e_ead_def = (ead_path_t + add) * pmf
    return np.sum(e_ead_def, axis=1)  # expected EAD at default

def scenario_policy_rate(snap, scen):
    if "macro_policy_rate" in snap.columns:
        # Ortalama policy rate kullan (tüm satırlar için tek değer)
        return snap["macro_policy_rate"].mean()
    return SCENARIOS[scen]["policy"]

def get_segment_series(snap):
    if "segment" in snap.columns:
        return snap["segment"].astype(str)
    if "product" in snap.columns:
        return snap["product"].astype(str)
    return pd.Series(["Unknown"]*len(snap), index=snap.index)

def staging_rules(snap, panel=None):
    """
    Stage 1: 12m ECL
    Stage 2: Lifetime (SICR)
    Stage 3: Default
    SICR heuristics:
      • PD level > 5%  OR
      • Relative jump: PD_cal / TTC > 1.5  OR
      • 30dpd+ backstop (panel varsa)
    """
    pd_cal = snap.get("pd_calibrated", snap.get("pd12_last", 0.03)).to_numpy()
    ttc = snap.get("pd12_last", pd_cal).to_numpy()
    rel_jump = (np.clip(pd_cal,1e-6,1.0)/np.clip(ttc,1e-6,1.0)) > 1.5
    high_level = pd_cal > 0.05

    backstop_30 = np.zeros(len(snap), dtype=bool)
    default_flag = np.zeros(len(snap), dtype=bool)
    if panel is not None and {"account_id","state"}.issubset(panel.columns):
        sev_map = {"current":0,"30dpd":1,"60dpd":2,"90dpd":3,"default":4,"closed":0}
        p = panel[["account_id","state"]].copy()
        p["sev"] = p["state"].map(sev_map).fillna(0).astype(int)
        maxsev = p.groupby("account_id")["sev"].max()
        backstop_30 = snap["account_id"].map(maxsev).fillna(0).to_numpy() >= 1
        default_flag = snap["account_id"].map(maxsev).fillna(0).to_numpy() >= 4
    else:
        # fallback: stage_last/target_default varsa kullan
        default_flag = ((snap.get("stage_last",0).to_numpy() >= 3) |
                        (snap.get("target_default",0).to_numpy()==1))

    stage = np.where(default_flag, 3,
             np.where(high_level | rel_jump | backstop_30, 2, 1)).astype(np.int8)
    
    # DataFrame'e stage sütununu ekle ve DataFrame'i geri döndür
    snap = snap.copy()
    snap['stage'] = stage
    return snap

# ---------------------------
# Core ECL per scenario
# ---------------------------
def ecl_per_scenario(snap: pd.DataFrame, panel: pd.DataFrame, scen_name: str, months=MONTHS_HORIZON):
    unemp = SCENARIOS[scen_name]["unemp"]
    gdp   = SCENARIOS[scen_name]["gdp"]

    seg = get_segment_series(snap)
    is_rev = seg.isin(REV_NAMES).to_numpy()

    # PD12 base
    pd12_base = snap.get("pd12_last", snap.get("pd_calibrated", 0.03)).to_numpy()
    # hazard base + macro adj
    h_month = scenario_adjust_hazard(pd12_base, unemp, gdp)                 # (n,)
    # seasoning shape
    season = snap.get("seasoning_m", None)
    shape = seasoning_shape(months, season.to_numpy() if season is not None else None)
    # monthly hazard path
    mhaz = np.clip(h_month[:,None] * (shape if shape.ndim==1 else 1.0), 1e-7, 0.999) if season is None \
           else np.clip(h_month[:,None] * shape, 1e-7, 0.999)                          # (n,T)

    # PMF(default at t)
    pmf = default_month_pmf(mhaz)                                                      # (n,T)

    # Discount curve
    # Per-account EIR (fallback to scenario policy)
    eir = snap.get("eir", pd.Series(np.nan, index=snap.index)).to_numpy()
    policy = scenario_policy_rate(snap, scen_name)
    eff_rate = np.where(np.isfinite(eir), eir, policy + DISCOUNT_SPREAD)
    disc = np.vstack([monthly_discount_factors(r, months) for r in eff_rate])

    # EAD path
    ead0 = snap.get("ead_first", snap.get("ead_last", 0.0)).to_numpy()
    risk_score = snap.get("risk_score", pd.Series(np.full(len(snap), 50), index=snap.index)).to_numpy()
    ead_t = ead_path(ead0, months, is_rev, risk_score=risk_score)                      # (n,T)

    # Undrawn & draw at default
    limit = snap.get("limit", snap.get("ead_last", 0.0)).to_numpy()
    undrawn = np.maximum(limit - snap.get("ead_last", 0.0).to_numpy(), 0.0)
    ccf = snap.get("ccf_product", pd.Series(np.full(len(snap), DRAW_ON_DEFAULT_REV), index=snap.index)).to_numpy()
    e_ead_def = revolving_draw_expectation(ead_t, undrawn, is_rev.astype(float), pmf, ccf=ccf)  # (n,)

    # LGD path
    # LGD: prefer lgd_est; if secured & ltv available, adjust; else fallback to lgd
    lgd_field = snap.get("lgd_est", snap.get("lgd", None))
    if lgd_field is None:
        lgd0 = np.full(len(snap), 0.45)  # Default LGD for all accounts
    else:
        lgd0 = np.clip(lgd_field.to_numpy(), 0.01, 0.99)
    # Additional LTV-based dampening if info present
    if "secured" in snap.columns:
        secured = snap["secured"].to_numpy()
        ltv_col = "ltv_last" if "ltv_last" in snap.columns else ("ltv" if "ltv" in snap.columns else None)
        if ltv_col is not None:
            ltv = np.clip(snap[ltv_col].fillna(1.0).to_numpy(), 0.1, 2.0)
            lgd0 = np.where(secured==1, np.clip(lgd0 * (0.5 + 0.8*ltv), 0.01, 0.99), lgd0)
    lgd_t = lgd_curve(lgd0, months, unemp, gdp)                                        # (n,T)

    # Expected monthly loss (base)
    el_months = pmf * ead_t * lgd_t * disc                                             # (n,T)
    el_base = np.sum(el_months, axis=1)

    # Revolving draw correction (use mean LGD & discount as approximation)
    lgd_mean = np.mean(lgd_t, axis=1)
    disc_mean = np.mean(disc, axis=1)
    el_corr = (e_ead_def - np.sum(pmf*ead_t, axis=1)) * lgd_mean * disc_mean
    el_lifetime = np.maximum(el_base + el_corr, 0.0)

    # 12-month ECL
    T12 = min(12, months)
    el_12m = np.sum(pmf[:,:T12] * ead_t[:,:T12] * lgd_t[:,:T12] * disc[:,:T12], axis=1)

    # Stage 3 accrual stop (bilgi amaçlı): post-default faiz yok sayılır; hesaplanan EL buna gömülü
    return el_12m, el_lifetime

def run_all_scenarios(snap: pd.DataFrame, panel: pd.DataFrame, months=MONTHS_HORIZON):
    scen_res = {}
    for s in SCENARIOS.keys():
        el12, ellife = ecl_per_scenario(snap, panel, s, months=months)
        scen_res[s] = {"weight": SCENARIOS[s]["weight"], "el12": el12, "ellife": ellife}
    # weighted
    el12_w = sum(v["weight"]*v["el12"] for v in scen_res.values())
    elli_w = sum(v["weight"]*v["ellife"] for v in scen_res.values())
    return scen_res, el12_w, elli_w

# ---------------------------
# Main
# ---------------------------
def main():
    snap, panel = load_inputs()

    # Minimal sanity / fallbacks
    if "account_id" not in snap.columns:
        snap = snap.reset_index().rename(columns={"index":"account_id"})
    if "segment" not in snap.columns and "product" in snap.columns:
        snap["segment"] = snap["product"].astype(str)
    if "segment" not in snap.columns:
        snap["segment"] = "Unknown"

    # Staging (SICR + backstops)
    snap = staging_rules(snap, panel)

    # Scenario ECLs (weighted)
    scen_out, el12_w, elli_w = run_all_scenarios(snap, panel, months=MONTHS_HORIZON)
    snap["ecl12_weighted"] = el12_w
    snap["ecl_lifetime_weighted"] = elli_w

    # Stage mapping (use 'stage' if available)
    stage_col = "stage" if "stage" in snap.columns else ("stage_calc" if "stage_calc" in snap.columns else None)
    if stage_col is not None:
        snap["ecl_stage"] = np.where(
            snap[stage_col]==1, snap["ecl12_weighted"], snap["ecl_lifetime_weighted"]
        )
    else:
        snap["ecl_stage"] = snap["ecl_lifetime_weighted"]

    # Save account level
    cols_keep = ["account_id","segment","stage_calc",
                 "pd_calibrated","pd12_last","lifetime_pd_last",
                 "ead_first","ead_last","limit",
                 "lgd_est","risk_score","risk_band",
                 "ecl12_weighted","ecl_lifetime_weighted","ecl_stage"]
    cols_exist = [c for c in cols_keep if c in snap.columns]
    acc = snap[cols_exist].copy()
    acc.to_parquet(ECL_DIR/"account_ecl.parquet", index=False)

    # Summaries
    def _sum(df, col): return float(np.nansum(df[col].to_numpy())) if col in df.columns else 0.0
    total_ead = _sum(snap, "ead_last")
    total_el12 = _sum(snap, "ecl12_weighted")
    total_elli = _sum(snap, "ecl_lifetime_weighted")

    cov12 = total_el12 / max(total_ead, 1e-9)
    covlife = total_elli / max(total_ead, 1e-9)

    # by segment
    if "segment" in snap.columns:
        seg_sum = snap.groupby("segment").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        seg_sum["cov12"] = seg_sum["ecl12"]/seg_sum["ead"].replace(0, np.nan)
        seg_sum["covlife"] = seg_sum["ecllife"]/seg_sum["ead"].replace(0, np.nan)
        seg_sum.to_csv(ECL_DIR/"summary_by_segment.csv", index=False, encoding="utf-8")

    # by stage
    stg_key = "stage" if "stage" in snap.columns else ("stage_calc" if "stage_calc" in snap.columns else None)
    stg_sum = snap.groupby(stg_key).agg(
        ead=("ead_last","sum"),
        ecl12=("ecl12_weighted","sum"),
        ecllife=("ecl_lifetime_weighted","sum"),
        cnt=("account_id","count")
    ).reset_index()
    if "stage" not in stg_sum.columns and stg_key is not None:
        stg_sum = stg_sum.rename(columns={stg_key:"stage"})
    stg_sum["cov12"] = stg_sum["ecl12"]/stg_sum["ead"].replace(0, np.nan)
    stg_sum["covlife"] = stg_sum["ecllife"]/stg_sum["ead"].replace(0, np.nan)
    stg_sum.to_csv(ECL_DIR/"summary_by_stage.csv", index=False, encoding="utf-8")

    # by product (if available)
    if "product" in snap.columns:
        prod_sum = snap.groupby("product").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        prod_sum["cov12"] = prod_sum["ecl12"]/prod_sum["ead"].replace(0, np.nan)
        prod_sum["covlife"] = prod_sum["ecllife"]/prod_sum["ead"].replace(0, np.nan)
        prod_sum.to_csv(ECL_DIR/"summary_by_product.csv", index=False, encoding="utf-8")

    # by secured flag (if available)
    if "secured" in snap.columns:
        sec_sum = snap.groupby("secured").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        sec_sum["cov12"] = sec_sum["ecl12"]/sec_sum["ead"].replace(0, np.nan)
        sec_sum["covlife"] = sec_sum["ecllife"]/sec_sum["ead"].replace(0, np.nan)
        sec_sum.to_csv(ECL_DIR/"summary_by_secured.csv", index=False, encoding="utf-8")

    # by LTV band (if available)
    ltv_col = "ltv_last" if "ltv_last" in snap.columns else ("ltv" if "ltv" in snap.columns else None)
    if ltv_col is not None:
        bins = [-1e9, 0.6, 0.8, 1.0, 1.2, 1e9]
        labels = ["<0.6","0.6-0.8","0.8-1.0","1.0-1.2",">1.2"]
        snap["ltv_band"] = pd.cut(snap[ltv_col].astype(float), bins=bins, labels=labels)
        ltv_sum = snap.groupby("ltv_band").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        ltv_sum["cov12"] = ltv_sum["ecl12"]/ltv_sum["ead"].replace(0, np.nan)
        ltv_sum["covlife"] = ltv_sum["ecllife"]/ltv_sum["ead"].replace(0, np.nan)
        ltv_sum.to_csv(ECL_DIR/"summary_by_ltv_band.csv", index=False, encoding="utf-8")

    # by scenario (totals)
    scen_rows = []
    for s, v in scen_out.items():
        scen_rows.append({
            "scenario": s,
            "weight": v["weight"],
            "ecl12": float(np.sum(v["el12"])),
            "ecl_life": float(np.sum(v["ellife"]))
        })
    pd.DataFrame(scen_rows).to_csv(ECL_DIR/"summary_by_scenario.csv", index=False, encoding="utf-8")

    # portfolio summary json
    summary = {
        "months_horizon": MONTHS_HORIZON,
        "total_ead": total_ead,
        "total_ecl12_w": total_el12,
        "total_ecllife_w": total_elli,
        "coverage_12m": cov12,
        "coverage_life": covlife,
        "scenarios": {k: {"weight": v["weight"]} for k,v in SCENARIOS.items()}
    }
    (ECL_DIR/"portfolio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Export Excel workbook (best-effort)
    try:
        xls = ECL_DIR/"portfolio_detail.xlsx"
        with pd.ExcelWriter(xls) as writer:
            if "seg_sum" in locals(): seg_sum.to_excel(writer, sheet_name="by_segment", index=False)
            stg_sum.to_excel(writer, sheet_name="by_stage", index=False)
            pd.DataFrame(scen_rows).to_excel(writer, sheet_name="by_scenario", index=False)
            if "prod_sum" in locals(): prod_sum.to_excel(writer, sheet_name="by_product", index=False)
            if "sec_sum" in locals(): sec_sum.to_excel(writer, sheet_name="by_secured", index=False)
            if "ltv_sum" in locals(): ltv_sum.to_excel(writer, sheet_name="by_ltv_band", index=False)
    except Exception:
        pass

    # small preview for business
    prev_cols = [c for c in ["account_id","segment","stage_calc","pd_calibrated","ead_last","ecl_stage","risk_band"] if c in acc.columns]
    acc.sort_values("ecl_stage", ascending=False).head(5000)[prev_cols].to_csv(ECL_DIR/"account_ecl_preview.csv", index=False, encoding="utf-8")

    print("== IFRS9 ECL COMPLETED ==")
    print(f"Total EAD:        {total_ead:,.0f}")
    print(f"Total 12m ECL(w): {total_el12:,.0f}  | Coverage: {cov12:.4f}")
    print(f"Total LT  ECL(w): {total_elli:,.0f} | Coverage: {covlife:.4f}")
    print(f"Outputs -> {ECL_DIR}")

# ----------------------
# Pipeline wrapper function
# ----------------------
def run_ecl(scored_df, panel, outdir="artifacts/ecl"):
    """
    Pipeline için ECL hesaplama fonksiyonu: dosyaları outdir altına yazar.
    """
    # Hedef klasör
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("🔹 ECL hesaplamaları başlıyor...")

    # 1) Staging
    snap = staging_rules(scored_df.copy(), panel)
    # Segment fallback: segment yoksa product'tan türet, o da yoksa Unknown ata
    if "segment" not in snap.columns:
        if "product" in snap.columns:
            snap["segment"] = snap["product"].astype(str)
        else:
            snap["segment"] = "Unknown"

    # 2) Senaryolar
    scen_out, el12_w, elli_w = run_all_scenarios(snap, panel)
    snap["ecl12_weighted"] = el12_w
    snap["ecl_lifetime_weighted"] = elli_w
    stage_col = "stage" if "stage" in snap.columns else ("stage_calc" if "stage_calc" in snap.columns else None)
    if stage_col is not None:
        snap["ecl_stage"] = np.where(snap[stage_col]==1, snap["ecl12_weighted"], snap["ecl_lifetime_weighted"]) 
    else:
        snap["ecl_stage"] = snap["ecl_lifetime_weighted"]

    # 3) Hesap bazlı çıktı
    cols_keep = [
        "account_id","segment","stage_calc",
        "pd_calibrated","pd12_last","lifetime_pd_last",
        "ead_first","ead_last","limit",
        "lgd_est","risk_score","risk_band",
        "ecl12_weighted","ecl_lifetime_weighted","ecl_stage"
    ]
    cols_exist = [c for c in cols_keep if c in snap.columns]
    acc = snap[cols_exist].copy()
    acc.to_parquet(outdir/"account_ecl.parquet", index=False)

    # 4) Özetler
    def _sum(df, col):
        return float(np.nansum(df[col].to_numpy())) if col in df.columns else 0.0
    total_ead = _sum(snap, "ead_last")
    total_el12 = _sum(snap, "ecl12_weighted")
    total_elli = _sum(snap, "ecl_lifetime_weighted")
    cov12 = total_el12 / max(total_ead, 1e-9)
    covlife = total_elli / max(total_ead, 1e-9)

    if "segment" in snap.columns:
        seg_sum = snap.groupby("segment").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        seg_sum["cov12"] = seg_sum["ecl12"]/seg_sum["ead"].replace(0, np.nan)
        seg_sum["covlife"] = seg_sum["ecllife"]/seg_sum["ead"].replace(0, np.nan)
        seg_sum.to_csv(outdir/"summary_by_segment.csv", index=False, encoding="utf-8")

    stg_key = "stage" if "stage" in snap.columns else ("stage_calc" if "stage_calc" in snap.columns else None)
    stg_sum = snap.groupby(stg_key).agg(
        ead=("ead_last","sum"),
        ecl12=("ecl12_weighted","sum"),
        ecllife=("ecl_lifetime_weighted","sum"),
        cnt=("account_id","count")
    ).reset_index()
    if "stage" not in stg_sum.columns and stg_key is not None:
        stg_sum = stg_sum.rename(columns={stg_key:"stage"})
    stg_sum.to_csv(outdir/"summary_by_stage.csv", index=False, encoding="utf-8")

    # by product (if available)
    if "product" in snap.columns:
        prod_sum = snap.groupby("product").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        prod_sum["cov12"] = prod_sum["ecl12"]/prod_sum["ead"].replace(0, np.nan)
        prod_sum["covlife"] = prod_sum["ecllife"]/prod_sum["ead"].replace(0, np.nan)
        prod_sum.to_csv(outdir/"summary_by_product.csv", index=False, encoding="utf-8")

    # by secured flag (if available)
    if "secured" in snap.columns:
        sec_sum = snap.groupby("secured").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        sec_sum["cov12"] = sec_sum["ecl12"]/sec_sum["ead"].replace(0, np.nan)
        sec_sum["covlife"] = sec_sum["ecllife"]/sec_sum["ead"].replace(0, np.nan)
        sec_sum.to_csv(outdir/"summary_by_secured.csv", index=False, encoding="utf-8")

    # by LTV band (if available)
    ltv_col = "ltv_last" if "ltv_last" in snap.columns else ("ltv" if "ltv" in snap.columns else None)
    if ltv_col is not None:
        bins = [-1e9, 0.6, 0.8, 1.0, 1.2, 1e9]
        labels = ["<0.6","0.6-0.8","0.8-1.0","1.0-1.2",">1.2"]
        snap["ltv_band"] = pd.cut(snap[ltv_col].astype(float), bins=bins, labels=labels)
        ltv_sum = snap.groupby("ltv_band").agg(
            ead=("ead_last","sum"),
            ecl12=("ecl12_weighted","sum"),
            ecllife=("ecl_lifetime_weighted","sum"),
            cnt=("account_id","count")
        ).reset_index()
        ltv_sum["cov12"] = ltv_sum["ecl12"]/ltv_sum["ead"].replace(0, np.nan)
        ltv_sum["covlife"] = ltv_sum["ecllife"]/ltv_sum["ead"].replace(0, np.nan)
        ltv_sum.to_csv(outdir/"summary_by_ltv_band.csv", index=False, encoding="utf-8")

    scen_rows = []
    for s, v in scen_out.items():
        scen_rows.append({
            "scenario": s,
            "weight": v["weight"],
            "ecl12": float(np.sum(v["el12"])),
            "ecl_life": float(np.sum(v["ellife"]))
        })
    pd.DataFrame(scen_rows).to_csv(outdir/"summary_by_scenario.csv", index=False, encoding="utf-8")

    summary = {
        "months_horizon": MONTHS_HORIZON,
        "total_ead": total_ead,
        "total_ecl12_w": total_el12,
        "total_ecllife_w": total_elli,
        "coverage_12m": cov12,
        "coverage_life": covlife,
        "scenarios": {k: {"weight": v["weight"]} for k,v in SCENARIOS.items()}
    }
    (outdir/"portfolio_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Export Excel workbook (best-effort)
    try:
        xls = outdir/"portfolio_detail.xlsx"
        with pd.ExcelWriter(xls) as writer:
            if "seg_sum" in locals(): seg_sum.to_excel(writer, sheet_name="by_segment", index=False)
            stg_sum.to_excel(writer, sheet_name="by_stage", index=False)
            pd.DataFrame(scen_rows).to_excel(writer, sheet_name="by_scenario", index=False)
            if "prod_sum" in locals(): prod_sum.to_excel(writer, sheet_name="by_product", index=False)
            if "sec_sum" in locals(): sec_sum.to_excel(writer, sheet_name="by_secured", index=False)
            if "ltv_sum" in locals(): ltv_sum.to_excel(writer, sheet_name="by_ltv_band", index=False)
    except Exception:
        pass

    prev_cols = [c for c in ["account_id","segment","stage_calc","pd_calibrated","ead_last","ecl_stage","risk_band"] if c in acc.columns]
    acc.sort_values("ecl_stage", ascending=False).head(5000)[prev_cols].to_csv(outdir/"account_ecl_preview.csv", index=False, encoding="utf-8")

    print(f"✅ ECL hesaplamaları tamamlandı: {outdir}")
    return scen_out, acc

if __name__ == "__main__":
    main()
