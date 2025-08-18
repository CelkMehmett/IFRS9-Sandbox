# ===============================================================
# Credit Risk Sandbox — FULL SCRIPT (v5.1)
# End-to-end: Synthetic/User Ingestion → PIT/TTC PD → Markov chain
# → EAD/LGD/Recovery (stochastic) → IFRS9 Stage (12m vs Lifetime ECL)
# → EIR discount → Tail risk (t-copula) → Stage 3 interest policy
# → PD model (GBM / LightGBM+Optuna HPO, monotonic) + Calibrations
# → Rating (10’lu) + Grade PD calibration → Stress EL + Monte Carlo VaR/ES
# → Diagnostics (ROC, Calibration, Lift, CAP, PSI) → HTML/PDF report
# → Ops/Governance: parquet export, scenario grid, vintage, roll-rate, drift alerts, dashboard (JSON)
# ===============================================================

import os, sys, io, json, argparse, warnings, base64, gzip, random, uuid, shutil
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional libs
try:
    import lightgbm as lgb
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False
try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False
try:
    import pdfkit
    HAVE_PDFKIT = True
except Exception:
    HAVE_PDFKIT = False
try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, average_precision_score, auc as sk_auc
)

# ===================== CONFIG =====================

@dataclass
class SegmentCfg:
    share: float
    pit_pd: float
    ttc_pd: float
    lgd_sec: float
    lgd_unsec: float
    ead_min: float
    ead_max: float
    rate_ann: float
    is_revolving: bool
    rho_pd: float
    rho_lgd: float
    rho_ead: float
    collateral_p: Dict[str,float]

@dataclass
class SICRCfg:
    abs_threshold: float = 0.05
    rel_multiplier: float = 3.0

@dataclass
class MacroCfg:
    mu: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.09, 0.03, 0.33]))  # gdp, unemp, hpi, policy
    phi: np.ndarray = field(default_factory=lambda: np.array([[0.62,0.12,-0.06,0.07],
                                                             [0.06,0.74,-0.02,0.06],
                                                             [0.12,0.02, 0.66,0.04],
                                                             [0.02,0.06, 0.01,0.72]]))
    sig: np.ndarray = field(default_factory=lambda: np.array([[0.010,0.004,0.006,0.003],
                                                             [0.004,0.010,0.003,0.003],
                                                             [0.006,0.003,0.012,0.003],
                                                             [0.003,0.003,0.003,0.010]]))
    base_rf_annual: float = 0.045
    liq_spread_annual: float = 0.015
    forward_policy_beta: float = 0.6

@dataclass
class ModelCfg:
    horizon_m: int = 60
    asof: str = "2025-01-01"
    seed: int = 42
    oot_train_min_back: int = 24
    oot_train_max_back: int = 59
    oot_valid_min_back: int = 12
    oot_valid_max_back: int = 23
    oot_test_min_back: int = 0
    oot_test_max_back: int = 11

@dataclass
class PDMapCfg:
    beta_pd: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Mortgage': {'gdp':-1.8,'unemp':2.0,'hpi':-1.2,'policy':0.8},
        'SME':      {'gdp':-2.4,'unemp':2.8,'hpi':-0.6,'policy':1.1},
        'Consumer': {'gdp':-1.6,'unemp':1.9,'hpi':-0.2,'policy':0.7}
    })

@dataclass
class StressCfg:
    scenarios: Dict[str, Tuple[float,float,float,float]] = field(default_factory=lambda: {
        "Baseline": (0.02, 0.09, 0.03, 0.33),
        "Adverse":  (0.00, 0.11, 0.01, 0.38),
        "Severe":   (-0.02, 0.14, -0.02, 0.43)
    })

@dataclass
class ExtraCfg:
    t_copula_nu: int = 5
    stage3_interest_policy: str = "stop"  # "stop" or "net"
    report_to_pdf: bool = False
    report_path_html: str = "reports/report.html"
    report_path_pdf: str  = "reports/report.pdf"
    mc_runs: int = 200

@dataclass
class SandboxCfg:
    seg: Dict[str, SegmentCfg]
    sicr: SICRCfg     = field(default_factory=SICRCfg)
    macro: MacroCfg   = field(default_factory=MacroCfg)
    model: ModelCfg   = field(default_factory=ModelCfg)
    pdmap: PDMapCfg   = field(default_factory=PDMapCfg)
    stress: StressCfg = field(default_factory=StressCfg)
    extra: ExtraCfg   = field(default_factory=ExtraCfg)

def default_config() -> SandboxCfg:
    return SandboxCfg(seg={
        'Mortgage': SegmentCfg(share=0.38,pit_pd=0.022,ttc_pd=0.018,lgd_sec=0.25,lgd_unsec=0.55,
                               ead_min=120_000, ead_max=3_000_000, rate_ann=0.22, is_revolving=False,
                               rho_pd=0.35, rho_lgd=0.25, rho_ead=0.15,
                               collateral_p={'property':0.85,'vehicle':0.03,'cash':0.02,'none':0.10}),
        'SME':      SegmentCfg(share=0.34,pit_pd=0.11, ttc_pd=0.095,lgd_sec=0.40,lgd_unsec=0.65,
                               ead_min=60_000, ead_max=4_000_000, rate_ann=0.28, is_revolving=False,
                               rho_pd=0.45, rho_lgd=0.35, rho_ead=0.25,
                               collateral_p={'property':0.35,'vehicle':0.20,'cash':0.10,'none':0.35}),
        'Consumer': SegmentCfg(share=0.28,pit_pd=0.065,ttc_pd=0.055,lgd_sec=0.50,lgd_unsec=0.75,
                               ead_min=2_000, ead_max=120_000, rate_ann=0.32, is_revolving=True,
                               rho_pd=0.25, rho_lgd=0.20, rho_ead=0.10,
                               collateral_p={'property':0.00,'vehicle':0.05,'cash':0.05,'none':0.90}),
    })

# ===================== UTILS =====================

STATE = ["CUR","D30","D60","D90","DEF","PREP"]

def seed_everywhere(seed:int):
    np.random.seed(seed); random.seed(seed)

def var1_paths(mu, phi, sig, T):
    y = np.zeros((T, len(mu))); y[0] = mu
    for t in range(1,T):
        eps = np.random.multivariate_normal(np.zeros(len(mu)), sig)
        y[t] = mu + phi @ (y[t-1]-mu) + eps
    return y

def seasoning_shape(m):
    return (0.9*np.exp(-((m-12)/12)**2) + 0.2/(1+0.02*m))

def dpd_of(s): return {0:0,1:30,2:60,3:90,4:180,5:0}[s]

def compute_eir_df(df: pd.DataFrame, cfg: SandboxCfg, H:int):
    r_m = df["segment"].map({k: v.rate_ann/12 for k, v in cfg.seg.items()}).values
    t = np.arange(H)[None, :]
    DF = (1.0 / (1.0 + r_m[:, None]) ) ** t
    DF[:,0] = 1.0
    return DF

def _safe_quantiles(arr, q):
    qs = np.quantile(arr, q)
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = np.nextafter(qs[i-1], np.inf)
    return qs

def quantile_thresholds(values: np.ndarray, n: int = 10) -> np.ndarray:
    values = np.asarray(values, float)
    qs = _safe_quantiles(values, np.linspace(0, 1, n+1))
    qs[0]  = qs[0]  - 1e-12
    qs[-1] = qs[-1] + 1e-12
    return qs

def feature_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, float)
    actual   = np.asarray(actual,   float)
    cuts = _safe_quantiles(expected, np.linspace(0,1,bins+1))
    if not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        return np.nan
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected),1)
    a = np.histogram(actual,   bins=cuts)[0] / max(len(actual),1)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((e-a)*np.log(e/a)))

def write_file(path:str, text:str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: f.write(text)

# ===================== TRANSITIONS =====================

BASE_TM = {
 'baseline': np.array([[0.958,0.022,0.000,0.000,0.004,0.016],
                       [0.280,0.585,0.090,0.000,0.025,0.020],
                       [0.060,0.150,0.585,0.140,0.045,0.020],
                       [0.000,0.000,0.070,0.670,0.210,0.050],
                       [0.000,0.000,0.000,0.000,1.000,0.000],
                       [0.000,0.000,0.000,0.000,0.000,1.000]]),
 'adverse':  np.array([[0.942,0.032,0.006,0.000,0.010,0.010],
                       [0.230,0.545,0.150,0.020,0.045,0.010],
                       [0.040,0.120,0.530,0.220,0.070,0.020],
                       [0.000,0.000,0.060,0.610,0.280,0.050],
                       [0.000,0.000,0.000,0.000,1.000,0.000],
                       [0.000,0.000,0.000,0.000,0.000,1.000]]),
 'severe':   np.array([[0.920,0.050,0.012,0.000,0.018,0.000],
                       [0.180,0.500,0.210,0.050,0.060,0.000],
                       [0.030,0.100,0.500,0.250,0.100,0.020],
                       [0.000,0.000,0.050,0.570,0.330,0.050],
                       [0.000,0.000,0.000,0.000,1.000,0.000],
                       [0.000,0.000,0.000,0.000,0.000,1.000]])
}

def pick_tm_from_macro(t:int, GDP:np.ndarray, UEM:np.ndarray, macro_mu:np.ndarray) -> np.ndarray:
    if (GDP[t] > -0.005) and (UEM[t] <= macro_mu[1]+0.01): return BASE_TM['baseline'].copy()
    if (GDP[t] > -0.02)  and (UEM[t] <= macro_mu[1]+0.03): return BASE_TM['adverse'].copy()
    return BASE_TM['severe'].copy()

# ===================== INGEST OR SYNTHETIC =====================

def ingest_or_generate_portfolio(cfg: SandboxCfg,
                                 user_df: Optional[pd.DataFrame]=None,
                                 n_total:int=40000) -> pd.DataFrame:
    if user_df is not None:
        cols = {c.lower(): c for c in user_df.columns}
        req = ["segment","income","score","ltv0","limit","util0"]
        for r in req: assert r in cols, f"Missing column: {r}"
        df = user_df.copy()
        for r in req:
            if r != cols[r]: df[r] = df[cols[r]]
        df["ead0"]   = df[cols["ead0"]] if "ead0" in cols else df["limit"]*df["util0"]
        df["tenor_m"]= df[cols["tenor_m"]] if "tenor_m" in cols else np.where(df["segment"].str.lower()=="consumer",999,60)
        df["secured"]= df[cols["secured"]].astype(int) if "secured" in cols else np.where(df["segment"].str.lower()=="mortgage",1,0)
        df["collateral_type"] = df[cols["collateral_type"]] if "collateral_type" in cols else np.where(df["secured"]==1,"property","none")
        asof = pd.to_datetime(cfg.model.asof)
        df["orig_month_back"] = np.random.randint(0, cfg.model.horizon_m, size=len(df))
        df["orig_date"] = asof - pd.to_timedelta(df["orig_month_back"]*30, unit="D")
        df["seasoning0"] = np.minimum(cfg.model.horizon_m-1, df["orig_month_back"])
        df["pit_pd_base"] = df["segment"].map({k:v.pit_pd for k,v in cfg.seg.items()})
        df["ttc_pd"]      = df["segment"].map({k:v.ttc_pd for k,v in cfg.seg.items()})
        return df.reset_index(drop=True)

    sizes = [int(n_total*cfg.seg[s].share) for s in cfg.seg]
    seg_list = np.concatenate([[s]*n for s,n in zip(cfg.seg.keys(), sizes)])
    df = pd.DataFrame({"segment": seg_list})
    asof = pd.to_datetime(cfg.model.asof)
    df["orig_month_back"] = np.random.randint(0, cfg.model.horizon_m, size=len(df))
    df["orig_date"] = asof - pd.to_timedelta(df["orig_month_back"]*30, unit="D")
    df["seasoning0"] = np.minimum(cfg.model.horizon_m-1, df["orig_month_back"])
    df["income"] = np.where(df.segment=="Mortgage", np.random.lognormal(10.5,0.42,len(df)),
                            np.where(df.segment=="SME", np.random.lognormal(10.2,0.55,len(df)),
                                     np.random.lognormal(9.4,0.55,len(df))))
    df["score"]  = np.clip(np.where(df.segment=="Mortgage", np.random.normal(730,50,len(df)),
                            np.where(df.segment=="SME", np.random.normal(675,65,len(df)),
                                     np.random.normal(645,70,len(df)))), 300, 900)
    df["ltv0"] = np.clip(np.where(df.segment=="Mortgage", np.random.normal(0.64,0.12,len(df)),
                                  np.where(df.segment=="SME", np.random.normal(0.58,0.15,len(df)),
                                           np.random.normal(0.52,0.18,len(df)))), 0.05, 1.8)
    df["limit"] = np.array([np.random.uniform(cfg.seg[s].ead_min, cfg.seg[s].ead_max) for s in df.segment])
    df["util0"]  = np.where(df.segment=="Mortgage", np.random.beta(10,2,len(df)),
                            np.where(df.segment=="SME", np.random.beta(3.0,4.0,len(df)),
                                     np.random.beta(2.3,4.2,len(df))))
    df["ead0"]   = df["limit"]*df["util0"]
    df["tenor_m"]= np.where(df.segment=="Consumer", 999, np.random.randint(18, 144, size=len(df)))
    df["secured"]= np.where(df.segment=="Mortgage", np.random.binomial(1,0.92,len(df)),
                            np.where(df.segment=="SME", np.random.binomial(1,0.62,len(df)),
                                     np.random.binomial(1,0.22,len(df))))
    coll = []
    for i in range(len(df)):
        s = df.segment.iat[i]; ps = cfg.seg[s].collateral_p
        if df.secured.iat[i]==0: coll.append("none")
        else: coll.append(np.random.choice(list(ps.keys()), p=list(ps.values())))
    df["collateral_type"] = coll
    df["pit_pd_base"] = df.segment.map({k:v.pit_pd for k,v in cfg.seg.items()})
    df["ttc_pd"]      = df.segment.map({k:v.ttc_pd for k,v in cfg.seg.items()})
    return df.reset_index(drop=True)

def ingest_or_generate_macro(cfg: SandboxCfg, user_macro_df: Optional[pd.DataFrame]=None) -> np.ndarray:
    if user_macro_df is not None:
        cols = {c.lower():c for c in user_macro_df.columns}
        req = ["gdp","unemp","hpi","policy"]
        for r in req: assert r in cols, f"Missing macro column: {r}"
        macro = user_macro_df[[cols["gdp"], cols["unemp"], cols["hpi"], cols["policy"]]].values
        H = cfg.model.horizon_m
        if macro.shape[0] < H:
            pad = np.tile(macro[-1:], (H - macro.shape[0], 1))
            macro = np.vstack([macro, pad])
        return macro[:H]
    return var1_paths(cfg.macro.mu, cfg.macro.phi, cfg.macro.sig, cfg.model.horizon_m)

# ===================== T-COPULA SHOCKS =====================

def t_copula_samples(R, nu, n):
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    Z = np.random.normal(size=(n, d)) @ L.T
    W = np.random.chisquare(df=nu, size=(n, 1))
    T = Z / np.sqrt(W/nu)
    if nu > 2: T = T / np.sqrt(nu/(nu-2))
    return T

def sample_segment_tcopula_shocks(df, cfg, nu=5):
    shocks = np.zeros((len(df), 3))
    for s, seg in cfg.seg.items():
        R = np.array([[1.0, seg.rho_pd, seg.rho_lgd],
                      [seg.rho_pd, 1.0, seg.rho_ead],
                      [seg.rho_lgd, seg.rho_ead, 1.0]])
        idx = (df.segment==s).values
        m = idx.sum()
        if m == 0: continue
        T = t_copula_samples(R, nu, m)
        shocks[idx,:] = T
    return shocks[:,0], shocks[:,1], shocks[:,2]

# ===================== PD HAZARD (TTC→PIT) =====================

def pit12(seg: str, ttc: float, t: int, z: float, cfg: SandboxCfg, GDP: np.ndarray, UEM: np.ndarray, HPI: np.ndarray, POL: np.ndarray, tail_scale=0.20):
    b = cfg.pdmap.beta_pd[seg]
    m = (b['gdp']*GDP[t] + b['unemp']*(UEM[t]-cfg.macro.mu[1]) + b['hpi']*HPI[t] + b['policy']*(POL[t]-cfg.macro.mu[3]))
    return np.clip(ttc * (1 + m) * (1 + tail_scale*z), 1e-5, 0.95)

# ===================== CORE PATHS =====================

def build_hazards(df, cfg, MACRO, shock_pd, tail_scale=0.30):
    GDP, UEM, HPI, POL = MACRO.T
    H = cfg.model.horizon_m
    MH = np.zeros((len(df), H))
    for i, row in df.iterrows():
        for t in range(H):
            p12 = pit12(row["segment"], row["ttc_pd"], t, shock_pd[i], cfg, GDP, UEM, HPI, POL, tail_scale=tail_scale)
            h   = 1 - (1 - p12)**(1/12)
            MH[i,t] = h * (0.85 + 0.15*seasoning_shape(np.clip(row["seasoning0"] + t, 0, 240)))
    return MH

def simulate_chains(df, cfg, MACRO, MH):
    GDP, UEM = MACRO[:,0], MACRO[:,1]
    H = cfg.model.horizon_m
    CHAINS = np.zeros((len(df),H), dtype=int)
    for i in range(len(df)):
        s = 0; high_sup_end = -1
        for t in range(1, H):
            Tm = pick_tm_from_macro(t, GDP, UEM, cfg.macro.mu).copy()
            seg = df.segment.iat[i]
            if seg == "SME":
                Tm[0,1] = min(0.08, Tm[0,1]*1.2); Tm[0,0] = 1 - (Tm[0,1]+Tm[0,4]+Tm[0,5])
                Tm[3,4] = min(0.60, Tm[3,4]*1.15); Tm[3,3] = 1 - (Tm[3,2]+Tm[3,4]+Tm[3,5])
            if seg == "Mortgage":
                Tm[0,5] = min(0.08, Tm[0,5]*1.25); Tm[0,0] = 1 - (Tm[0,1]+Tm[0,4]+Tm[0,5])
            Tm[s,4] = np.clip(Tm[s,4] + 0.70*MH[i,t], 0, 0.99)
            if s == 1 and 0 <= t < high_sup_end:
                Tm[1,0] = max(Tm[1,0]*0.85, 0.05); Tm[1,2] = min(Tm[1,2]*1.20, 0.45)
                Tm[1] = Tm[1]/Tm[1].sum()
            if s in [4,5]: CHAINS[i,t] = s; continue
            p = Tm[s]/Tm[s].sum()
            s = np.random.choice(len(STATE), p=p); CHAINS[i,t] = s
            if s == 0 and CHAINS[i,t-1] == 1: high_sup_end = t + 4
    return CHAINS

def ead_paths(df, cfg, MACRO, CHAINS):
    UEM, GDP = MACRO[:,1], MACRO[:,0]
    H = cfg.model.horizon_m
    EAD = np.zeros((len(df),H))
    for i in range(len(df)):
        seg = df.segment.iat[i]; r_m = cfg.seg[seg].rate_ann/12
        rev = cfg.seg[seg].is_revolving; limit = df.limit.iat[i]; tenor = df.tenor_m.iat[i]
        EAD[i,0] = df.ead0.iat[i]
        for t in range(1,H):
            s_prev, s_now = CHAINS[i,t-1], CHAINS[i,t]
            if s_prev==4: EAD[i,t] = EAD[i,t-1]; continue
            if rev:
                lim_adj = (1 - 0.10*(UEM[t]-cfg.macro.mu[1]) - 0.06*(df.score.iat[i]<600))
                limit_t = max(limit*lim_adj, 0.4*limit)
                drift = 0.06*(UEM[t]-cfg.macro.mu[1]) - 0.04*GDP[t] + np.random.normal(0,0.025)
                util  = np.clip(EAD[i,t-1]/limit_t + drift, 0.02, 1.20)
                ccf   = 0.80 + 0.12*(s_now in [1,2,3]) + 0.15*(s_now==3)
                EAD[i,t]= np.clip(limit_t*util*(1+0.25*(s_now==3))*(1+0.20*(s_now==4)), 0, limit_t*(1+ccf))
            else:
                rem = max(1, tenor - t)
                interest = max(0.0, EAD[i,t-1]*(r_m))
                ann = interest + (EAD[i,t-1]/max(rem,1))
                principal = max(0.0, ann - interest)
                EAD[i,t] = max(0.0, EAD[i,t-1] - principal)
                if s_now in [2,3]: EAD[i,t] *= (1 + 0.03*(s_now==2) + 0.07*(s_now==3))
            if s_now == 5: EAD[i,t] = 0.0
    return EAD

def draw_time_to_sale(collateral_type, uem_t, hpi_t):
    base = {"property": (np.log(12), 0.5),
            "vehicle":  (np.log(8),  0.6),
            "cash":     (np.log(3),  0.35),
            "none":     (np.log(6),  0.7)}.get(collateral_type, (np.log(6),0.7))
    mu, sigma = base
    tts = np.exp(np.random.normal(mu, sigma))
    tts = tts * (1 + 1.2*(uem_t - 0.09)) * (1 - 0.4*(hpi_t - 0.03))
    return int(max(1, min(36, round(tts))))

def stochastic_recovery_schedule(td, H, collateral_type, lgd_draw, uem_series, hpi_series):
    if td < 0 or td >= H-1: return np.zeros(H)
    K = min(6, 1 + np.random.geometric(p=0.55))
    tts = draw_time_to_sale(collateral_type, uem_series[td], hpi_series[td])
    rr_total = max(0.0, 1.0 - float(lgd_draw))
    costs = {"property":0.12, "vehicle":0.10, "cash":0.05, "none":0.06}.get(collateral_type, 0.08)
    rr_total *= (1 - costs)
    fail_prob = {"property":0.20, "vehicle":0.12, "cash":0.02, "none":0.08}.get(collateral_type, 0.10)
    fail = (np.random.rand() < fail_prob)
    rec = np.zeros(H)
    w = np.random.dirichlet(alpha=np.ones(K))
    month = td + tts
    for k in range(K):
        t = month + np.random.randint(-1, 2) + k*np.random.randint(0, 2)
        if t <= td: t = td + 1
        if t >= H: break
        if fail and k < 2: continue
        rec[t] += rr_total * w[k]
    return rec

def lgd_paths(df, cfg, MACRO, CHAINS, shock_lgd):
    HPI, UEM = MACRO[:,2], MACRO[:,1]
    H = cfg.model.horizon_m
    LGD = np.zeros((len(df),H))*np.nan
    REC = np.zeros((len(df),H))
    for i in range(len(df)):
        seg = df.segment.iat[i]; ct  = df.collateral_type.iat[i]
        secured = (ct!="none")
        base = cfg.seg[seg].lgd_sec if secured else cfg.seg[seg].lgd_unsec
        base *= (1 + 0.10*shock_lgd[i])
        if ct=="property": base *= (1 - 0.25*np.mean(HPI))
        if ct=="vehicle":  base *= 1.05
        if ct=="cash":     base *= 0.80
        base = np.clip(base, 0.02, 0.98)
        a,b = (base*30, (1-base)*30)
        lgd_draw = np.clip(np.random.beta(a,b), 0.02, 0.98)
        t_def = np.where(CHAINS[i]==4)[0]
        rec = np.zeros(H)
        if len(t_def)>0:
            td = t_def[0]
            rec = stochastic_recovery_schedule(td, H, ct, lgd_draw, UEM, HPI)
        LGD[i,:] = lgd_draw
        REC[i,:] = rec
    return LGD, REC

def incpd_from_chains(CHAINS):
    H = CHAINS.shape[1]
    INC_PD = np.zeros((CHAINS.shape[0],H))
    for i in range(CHAINS.shape[0]):
        alive = 1.0
        for t in range(1, H):
            if CHAINS[i,t] == 4 and alive > 0.5:
                INC_PD[i,t] = 1.0; alive = 0.0
    return INC_PD

def stage_ifrs9_labels(df, cfg, CHAINS, MACRO, shock_pd):
    H = CHAINS.shape[1]; GDP, UEM, HPI, POL = MACRO.T
    stage = []
    orig_pd = df["pit_pd_base"].values
    for i in range(len(df)):
        st = "Stage1"
        for t in range(H):
            pit_now = pit12(df.segment.iat[i], df.ttc_pd.iat[i], t, shock_pd[i], cfg, GDP, UEM, HPI, POL)
            cond_abs = pit_now >= cfg.sicr.abs_threshold
            cond_rel = pit_now/max(1e-6, orig_pd[i]) >= cfg.sicr.rel_multiplier
            cond_dpd = dpd_of(CHAINS[i,t]) >= 30
            if CHAINS[i,t]==4: st="Stage3"; break
            if cond_abs or cond_rel or cond_dpd: st="Stage2"
        stage.append(st)
    return np.array(stage)

def stage_based_ecl(df, EAD_PATH, LGD_PATH, CHAINS, DF_eir, horizon_12=12):
    INC_PD = incpd_from_chains(CHAINS)
    lgd = np.where(np.isnan(LGD_PATH), 0.5, LGD_PATH)
    loss_flow = EAD_PATH * lgd * INC_PD
    ECL_12m  = np.sum(loss_flow[:, :horizon_12] * DF_eir[:, :horizon_12], axis=1)
    ECL_LIFE = np.sum(loss_flow * DF_eir, axis=1)
    ECL_STAGE= np.where(df["stage_ifrs9"].values=="Stage1", ECL_12m, ECL_LIFE)
    return ECL_12m, ECL_LIFE, ECL_STAGE

def stage3_interest(df, CHAINS, EAD_PATH, LGD_PATH, cfg, policy="stop"):
    H = CHAINS.shape[1]
    r_m = df["segment"].map({k: v.rate_ann/12 for k, v in cfg.seg.items()}).values
    gross = np.zeros((len(df), H))
    stage3 = np.zeros((len(df), H))
    for i in range(len(df)):
        for t in range(1, H):
            gca_prev = EAD_PATH[i, t-1]
            gross[i, t] = gca_prev * r_m[i]
            if CHAINS[i, t] == 4 or np.any(CHAINS[i, :t] == 4):
                if policy == "stop":
                    stage3[i, t] = 0.0
                else:
                    lgd_pt = LGD_PATH[i, t] if t < H and not np.isnan(LGD_PATH[i, t]) else 0.45
                    nca_prev = max(0.0, gca_prev * (1 - lgd_pt))
                    stage3[i, t] = nca_prev * r_m[i]
            else:
                stage3[i, t] = gross[i, t]
    return gross, stage3

# ===================== MODEL & CALIBRATION =====================

def bucket_calibration(p, y, nb=10):
    cuts = _safe_quantiles(p, np.linspace(0,1,nb+1))
    cuts[0]-=1e-9; cuts[-1]+=1e-9
    idx = np.digitize(p, cuts)-1
    tgt = np.array([y[idx==b].mean() if (idx==b).any() else 0 for b in range(nb)])
    pred= np.array([p[idx==b].mean() if (idx==b).any() else 0 for b in range(nb)])
    scale = np.ones(nb); nz = pred>0; scale[nz] = tgt[nz]/pred[nz]
    return cuts, scale

def apply_bcal(p, cuts, scale):
    idx = np.digitize(p, cuts)-1; idx = np.clip(idx, 0, len(scale)-1)
    return np.clip(p*scale[idx], 1e-6, 1-1e-6)

def fit_lgbm_hpo(Xtr, ytr, Xva, yva, monotone_constraints=None, n_trials=30, timeout_s=180):
    if not (HAVE_LGBM and HAVE_OPTUNA): return None, None, None
    train_set = lgb.Dataset(Xtr, label=ytr); valid_set = lgb.Dataset(Xva, label=yva, reference=train_set)
    def objective(trial):
        params = {"objective":"binary","metric":"auc","verbosity":-1,"boosting_type":"gbdt",
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=8),
                  "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 800, step=50),
                  "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                  "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
                  "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
                  "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0)}
        if monotone_constraints is not None:
            params["monotone_constraints"] = monotone_constraints
        model = lgb.train(params, train_set, valid_sets=[valid_set], num_boost_round=300,
                          early_stopping_rounds=50, verbose_eval=False)
        preds = model.predict(Xva, num_iteration=model.best_iteration)
        return 1.0 - roc_auc_score(yva, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    best_params = study.best_trial.params
    params = {"objective":"binary","metric":"auc","verbosity":-1,"boosting_type":"gbdt", **best_params}
    if monotone_constraints is not None: params["monotone_constraints"] = monotone_constraints
    model = lgb.train(params, train_set, valid_sets=[valid_set], num_boost_round=1000,
                      early_stopping_rounds=100, verbose_eval=False)
    return model, params, study

# ===================== REPORT =====================

def _img_bytes_to_b64(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

def write_html_report(df, base_metrics, el12_df, html_path):
    os.makedirs(Path(html_path).parent, exist_ok=True)
    stage_counts = df["stage_ifrs9"].value_counts().to_dict()
    ecl_sum_12  = df.get("ECL_12m_EIR_t", pd.Series(dtype=float)).sum() if "ECL_12m_EIR_t" in df else df.get("ECL_12m_EIR", pd.Series(dtype=float)).sum()
    ecl_sum_lft = df.get("ECL_life_EIR_t", pd.Series(dtype=float)).sum() if "ECL_life_EIR_t" in df else df.get("ECL_lifetime_EIR", pd.Series(dtype=float)).sum()
    ecl_sum_stg = df.get("ECL_stage_EIR_t", pd.Series(dtype=float)).sum() if "ECL_stage_EIR_t" in df else df.get("ECL_stage_EIR", pd.Series(dtype=float)).sum()
    b64_roc  = _img_bytes_to_b64("reports/roc_curve.png")
    b64_cal  = _img_bytes_to_b64("reports/calibration_curve.png")
    b64_shap = _img_bytes_to_b64("reports/shap_summary.png")
    el12_html = el12_df.to_html(float_format=lambda x: f"{x:,.0f}")
    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Credit Risk Sandbox — Report</title>
<style>body{{font-family:Arial;margin:24px}}.kpi{{display:flex;gap:24px;flex-wrap:wrap}}.card{{border:1px solid #ddd;border-radius:12px;padding:12px 16px;min-width:220px}}img{{max-width:640px;border:1px solid #eee;border-radius:8px}}th,td{{border:1px solid #ddd;padding:6px 8px}}table{{border-collapse:collapse}}</style>
</head><body>
<h1>Credit Risk Sandbox — Summary</h1>
<h2>Model Metrics</h2>
<div class="kpi">
  <div class="card"><b>AUC (test)</b><br/>{base_metrics.get('AUC', float('nan')):.3f}</div>
  <div class="card"><b>KS</b><br/>{base_metrics.get('KS', float('nan')):.3f}</div>
  <div class="card"><b>Gini</b><br/>{base_metrics.get('Gini', float('nan')):.3f}</div>
  <div class="card"><b>Brier</b><br/>{base_metrics.get('Brier', float('nan')):.4f}</div>
  <div class="card"><b>PR-AUC</b><br/>{base_metrics.get('PR_AUC', float('nan')):.3f}</div>
  <div class="card"><b>PSI</b><br/>{base_metrics.get('PSI_total', float('nan')):.3f}</div>
</div>
<h2>IFRS 9 ECL (EIR)</h2>
<div class="kpi">
  <div class="card"><b>12m ECL</b><br/>{(ecl_sum_12 or 0):,.0f}</div>
  <div class="card"><b>Lifetime ECL</b><br/>{(ecl_sum_lft or 0):,.0f}</div>
  <div class="card"><b>Stage-based ECL</b><br/>{(ecl_sum_stg or 0):,.0f}</div>
</div>
<h3>Stage Distribution</h3>
<pre>{json.dumps(stage_counts, indent=2)}</pre>
<h2>12m Expected Loss by Scenario & Segment</h2>
{el12_html}
<h2>Diagnostics</h2>
<h3>ROC</h3>{("<img src='"+b64_roc+"'/>" if b64_roc else "<small>ROC missing</small>")}
<h3>Calibration</h3>{("<img src='"+b64_cal+"'/>" if b64_cal else "<small>Calibration missing</small>")}
<h3>SHAP</h3>{("<img src='"+b64_shap+"'/>" if b64_shap else "<small>SHAP missing</small>")}
<footer><small>Generated by Sandbox</small></footer>
</body></html>
"""
    write_file(html_path, html)
    return html_path

def export_pdf(html_path, pdf_path):
    if not HAVE_PDFKIT: return False
    try:
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        pdfkit.from_file(html_path, pdf_path); return True
    except Exception:
        return False

# ===================== OPS/GOV =====================

def export_parquet_or_gz(df:pd.DataFrame, path:Path):
    try:
        df.to_parquet(path, index=False); return str(path)
    except Exception:
        gz = path.with_suffix(".csv.gz")
        with gzip.open(gz, "wt", encoding="utf-8") as f: df.to_csv(f, index=False)
        return str(gz)

def scenario_uplift(pd_hat, g,u,h,pr, mu):
    return np.clip(pd_hat*(1 + -1.8*g + 2.2*(u-mu[1]) - 0.9*h + 0.9*(pr-mu[3])), 1e-6, 0.995)

def run_scenario_grid(pd_scores, seg_series, cfg, names, grid, lgd_by_seg, ead_by_seg):
    rows = []
    base_mu = cfg.macro.mu
    for name,(g,u,h,pr) in zip(names, grid):
        pd_s = scenario_uplift(pd_scores, g,u,h,pr, base_mu)
        pol_shift = 0.6*(pr - base_mu[3])
        df12_disc = float(np.exp(-np.sum((cfg.macro.base_rf_annual + cfg.macro.liq_spread_annual + pol_shift*np.ones(12))/12)))
        for s in cfg.seg:
            m = (seg_series==s)
            if m.sum()==0: continue
            el = float(pd_s[m].mean() * lgd_by_seg[s] * ead_by_seg[s] * df12_disc * m.sum())
            rows.append({"scenario":name,"segment":s,"EL12":el})
    return pd.DataFrame(rows)

# ===================== MAIN PIPELINE =====================

def run_pipeline_full(cfg: SandboxCfg, portfolio_df=None, macro_df=None, out_dir="outputs", rep_dir="reports"):
    seed_everywhere(cfg.model.seed)
    OUT, REP = Path(out_dir), Path(rep_dir)
    OUT.mkdir(exist_ok=True, parents=True); REP.mkdir(exist_ok=True, parents=True)

    # Data
    df = ingest_or_generate_portfolio(cfg, portfolio_df, n_total=40000)
    MACRO = ingest_or_generate_macro(cfg, macro_df)
    GDP, UEM, HPI, POL = MACRO.T
    H = cfg.model.horizon_m

    # Tail shocks
    shock_pd, shock_lgd, shock_ead = sample_segment_tcopula_shocks(df, cfg, nu=cfg.extra.t_copula_nu)
    df["shock_pd"], df["shock_lgd"], df["shock_ead"] = shock_pd, shock_lgd, shock_ead

    # Build hazards & chains
    MH = build_hazards(df, cfg, MACRO, shock_pd, tail_scale=0.30)
    CHAINS = simulate_chains(df, cfg, MACRO, MH)

    # EAD, LGD, Recovery
    EAD_PATH = ead_paths(df, cfg, MACRO, CHAINS)
    LGD_PATH, REC_PATH = lgd_paths(df, cfg, MACRO, CHAINS, shock_lgd)

    # IFRS9 Staging
    df["stage_ifrs9"] = stage_ifrs9_labels(df, cfg, CHAINS, MACRO, shock_pd)

    # ECL with EIR
    DF_eir = compute_eir_df(df, cfg, H)
    ECL_12m, ECL_LIFE, ECL_STAGE = stage_based_ecl(df, EAD_PATH, LGD_PATH, CHAINS, DF_eir, 12)
    df["ECL_12m_EIR_t"] = ECL_12m
    df["ECL_life_EIR_t"] = ECL_LIFE
    df["ECL_stage_EIR_t"]= ECL_STAGE

    # Stage 3 Interest Policy
    gross_int, stage3_int = stage3_interest(df, CHAINS, EAD_PATH, LGD_PATH, cfg, policy=cfg.extra.stage3_interest_policy)
    df["interest_gross_12m"] = gross_int[:,1:13].sum(1)
    df["interest_stage3_12m"] = stage3_int[:,1:13].sum(1)

    # Labels & OOT
    INC_PD = incpd_from_chains(CHAINS)
    label12 = (INC_PD[:,1:13].sum(1)>0).astype(int)
    tr = (df["orig_month_back"].between(cfg.model.oot_train_min_back, cfg.model.oot_train_max_back)).values
    va = (df["orig_month_back"].between(cfg.model.oot_valid_min_back, cfg.model.oot_valid_max_back)).values
    te = (df["orig_month_back"].between(cfg.model.oot_test_min_back,  cfg.model.oot_test_max_back)).values

    X = df[["income","score","ltv0","limit","ead0","secured"]].copy()
    X["seg_SME"] = (df.segment=="SME").astype(int)
    X["seg_CON"] = (df.segment=="Consumer").astype(int)
    y = label12
    Xtr,ytr = X[tr], y[tr]; Xva,yva = X[va], y[va]; Xte,yte = X[te], y[te]

    # Base GBM
    gbm = GradientBoostingClassifier(random_state=7).fit(Xtr,ytr)
    p_va = gbm.predict_proba(Xva)[:,1]; p_te = gbm.predict_proba(Xte)[:,1]
    auc_base = roc_auc_score(yte, p_te)

    # Sigmoid & Isotonic
    cal_sig = CalibratedClassifierCV(gbm, method="sigmoid", cv="prefit").fit(Xva,yva)
    p_te_sig = cal_sig.predict_proba(Xte)[:,1]; auc_sig = roc_auc_score(yte, p_te_sig)
    gbm_iso = GradientBoostingClassifier(random_state=11).fit(Xtr,ytr)
    cal_iso = CalibratedClassifierCV(gbm_iso, method="isotonic", cv=3).fit(Xva,yva)
    p_te_iso = cal_iso.predict_proba(Xte)[:,1]; auc_iso = roc_auc_score(yte, p_te_iso)

    # Bucket + segment align
    cuts_all, scale_all = bucket_calibration(p_va, yva, nb=10)
    p_te_bcal = apply_bcal(p_te, cuts_all, scale_all)
    p_te_seg = p_te_bcal.copy()
    df_te = df[te]
    for s in cfg.seg:
        m = (df_te.segment==s).values
        if m.sum()<50: continue
        tgt = yte[m].mean(); pred = p_te_bcal[m].mean()
        if pred>0:
            adj = np.clip(tgt/pred, 0.6, 1.6)
            p_te_seg[m] = np.clip(p_te_bcal[m]*adj, 1e-6, 1-1e-6)
    auc_segcal = roc_auc_score(yte, p_te_seg)
    fpr,tpr,_ = roc_curve(yte, p_te_seg); KS = float(np.max(tpr - fpr))
    Gini = float(2*roc_auc_score(yte, p_te_seg) - 1)
    brier = brier_score_loss(yte, p_te_seg)
    pr_auc = average_precision_score(yte, p_te_seg)
    PSI_total = feature_psi(p_va, p_te_seg, bins=10)

    # Optional SHAP
    if HAVE_SHAP:
        try:
            explainer = shap.TreeExplainer(gbm)
            shap_values = explainer.shap_values(Xte)
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            shap.summary_plot(shap_values, Xte, show=False); plt.tight_layout()
            Path(REP).mkdir(exist_ok=True, parents=True)
            plt.savefig(Path(REP)/"shap_summary.png", dpi=150); plt.close()
        except Exception:
            pass

    # Reliability & ROC plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        def calib_points(y_true, y_prob, n_bins=10):
            cuts = _safe_quantiles(y_prob, np.linspace(0,1,n_bins+1)); cuts[0]-=1e-9; cuts[-1]+=1e-9
            idx = np.digitize(y_prob, cuts)-1
            exp, obs = [], []
            for b in range(n_bins):
                m = idx==b
                if m.sum()==0: exp.append(np.nan); obs.append(np.nan)
                else: exp.append(y_prob[m].mean()); obs.append(y_true[m].mean())
            return np.array(exp), np.array(obs)
        exp_rate, obs_rate = calib_points(yte, p_te_seg, n_bins=10)
        plt.figure(); plt.plot(exp_rate, obs_rate, marker="o"); plt.plot([0,1],[0,1],"--")
        plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration (Test)")
        plt.tight_layout(); plt.savefig(Path(REP)/"calibration_curve.png", dpi=150); plt.close()
        fpr,tpr,_ = roc_curve(yte, p_te_seg)
        plt.figure(); plt.plot(fpr,tpr,label=f"AUC={sk_auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC (Test)")
        plt.tight_layout(); plt.savefig(Path(REP)/"roc_curve.png", dpi=150); plt.close()
    except Exception:
        pass

    # LightGBM + Optuna HPO (monotonic)
    mono = [-1, -1, 1, 0, 1, -1, 0, 0]
    best_model, best_params, study = fit_lgbm_hpo(Xtr, ytr, Xva, yva, monotone_constraints=mono, n_trials=30, timeout_s=180)
    if best_model is not None:
        p_te_hat = best_model.predict(Xte, num_iteration=best_model.best_iteration)
        AUC_lgbm = float(roc_auc_score(yte, p_te_hat))
    else:
        p_te_hat = p_te_seg
        AUC_lgbm = float("nan")

    # Stress EL (12m) by segment
    pd_use = p_te_hat
    EL_12M_by_seg = {}
    base_mu = cfg.macro.mu
    df12_disc = float(np.exp(-np.sum((cfg.macro.base_rf_annual + cfg.macro.liq_spread_annual + 0.6*(POL - POL.mean()))[:12]/12)))
    for name,(g,u,h,pr) in cfg.stress.scenarios.items():
        pd_s = scenario_uplift(pd_use, g,u,h,pr, base_mu)
        el = {}
        for s in cfg.seg:
            m = (df_te.segment==s).values
            if m.sum()==0: continue
            lgd_s = float(np.nanmean(LGD_PATH[:, :12])) if not np.isnan(LGD_PATH[:, :12]).all() else (cfg.seg[s].lgd_unsec*0.8 + cfg.seg[s].lgd_sec*0.2)
            ead_s = float(df.loc[df.segment==s,"ead0"].mean())
            el[s] = float(pd_s[m].mean()*lgd_s*ead_s*df12_disc)
        EL_12M_by_seg[name] = el
    EL_12M_by_seg_df = pd.DataFrame(EL_12M_by_seg)
    EL_12M_by_seg_df.to_csv(OUT/"el12_by_segment_scenarios_v4.csv", index=False)

    # Ratings (10’lu) & PSI
    seg_va = df.loc[Xva.index, "segment"]; seg_te = df.loc[Xte.index, "segment"]
    p_va_hat = p_va
    thr_map = {}
    for s in seg_va.unique():
        m = (seg_va==s).values
        thr_map[s] = quantile_thresholds(p_va_hat[m] if m.any() else p_va_hat, 10)
    grades_te = np.zeros_like(pd_use, dtype=int)
    for s, thr in thr_map.items():
        m = (seg_te==s).values
        grades_te[m] = np.clip(np.digitize(pd_use[m], thr), 1, 10)
    rating_rows = []
    for s in sorted(seg_te.unique()):
        sub = pd.DataFrame({"pd": pd_use[seg_te==s], "y": yte[seg_te==s], "g": grades_te[seg_te==s]})
        for g in range(1,11):
            sg = sub[sub.g==g]
            n = len(sg)
            obs = sg["y"].mean() if n>0 else np.nan
            exp = sg["pd"].mean() if n>0 else np.nan
            rating_rows.append({"segment": s, "grade": g, "n": n, "obs_pd": obs, "exp_pd": exp})
    pd.DataFrame(rating_rows).to_csv(OUT/"rating_table_test.csv", index=False)
    with open(OUT/"rating_thresholds.json","w") as f:
        json.dump({s:list(map(float,t)) for s,t in thr_map.items()}, f, indent=2)

    psi_df = pd.DataFrame([{"feature": c, "psi": feature_psi(np.array(Xva[c],float), np.array(Xte[c],float), bins=10)}
                           for c in X.columns])
    psi_df.to_csv(OUT/"feature_psi_valid_vs_test.csv", index=False)

    # Monte Carlo EL distribution + VaR/ES
    M = cfg.extra.mc_runs
    mu, phi, sig = cfg.macro.mu, cfg.macro.phi, cfg.macro.sig
    df_te_sub = df.loc[Xte.index]
    ead_by_seg = df_te_sub.groupby("segment")["ead0"].mean().to_dict()
    lgd_proxy = {s: (float(np.nanmean(LGD_PATH[df.segment==s, :12])) if not np.isnan(LGD_PATH[:, :12]).all()
                     else (cfg.seg[s].lgd_unsec*0.8 + cfg.seg[s].lgd_sec*0.2)) for s in cfg.seg}
    EL_port_mc = []
    for m in range(M):
        MACRO_m = var1_paths(mu, phi, sig, H)
        g,u,h,pr = MACRO_m[:12].mean(axis=0)
        pd_s = scenario_uplift(pd_use, g,u,h,pr, mu)
        POL_m = MACRO_m[:,3]
        df12_disc_m = float(np.exp(-np.sum((cfg.macro.base_rf_annual + cfg.macro.liq_spread_annual + 0.6*(POL_m - POL_m.mean()))[:12]/12)))
        el_sum = 0.0
        seg_series = df_te_sub["segment"].values
        for s in cfg.seg:
            mask = (seg_series==s)
            if mask.sum()==0: continue
            el_seg = float(pd_s[mask].mean() * lgd_proxy[s] * ead_by_seg[s] * df12_disc_m * mask.sum())
            el_sum += el_seg
        EL_port_mc.append(el_sum)
    EL_port_mc = np.array(EL_port_mc)
    pd.DataFrame({"EL12": EL_port_mc}).to_csv(OUT/"mc_el12_samples.csv", index=False)
    def var_es(samples: np.ndarray, alpha: float=0.99):
        var = float(np.quantile(samples, alpha)); es = float(samples[samples >= var].mean()) if (samples >= var).any() else var
        return var, es
    VaR95, ES95 = var_es(EL_port_mc, 0.95); VaR99, ES99 = var_es(EL_port_mc, 0.99)
    json.dump({"VaR95": VaR95, "ES95": ES95, "VaR99": VaR99, "ES99": ES99},
              open(OUT/"mc_el12_var_es.json","w"), indent=2)

    # Lift & CAP
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        df_test_sc = pd.DataFrame({"pd": pd_use, "y": yte}).sort_values("pd", ascending=False).reset_index(drop=True)
        df_test_sc["cum_obs"] = df_test_sc["y"].cumsum(); df_test_sc["cum_pct"] = (np.arange(len(df_test_sc))+1)/len(df_test_sc)
        total_bad = df_test_sc["y"].sum(); lift = (df_test_sc["cum_obs"]/max(total_bad,1)) / df_test_sc["cum_pct"]
        plt.figure(figsize=(7,5)); plt.plot(df_test_sc["cum_pct"], lift); plt.axhline(1.0, ls="--", lw=1)
        plt.xlabel("Kümülatif Nüfus"); plt.ylabel("Lift"); plt.title("Lift (Test)")
        plt.tight_layout(); plt.savefig(Path(REP)/"lift_curve.png", dpi=150); plt.close()
        ideal = np.concatenate([np.ones(int(total_bad)), np.zeros(len(df_test_sc)-int(total_bad))])
        ideal = np.cumsum(ideal)/max(total_bad,1)
        random_line = df_test_sc["cum_pct"].values
        cap = df_test_sc["cum_obs"].values / max(total_bad,1)
        plt.figure(figsize=(7,5))
        plt.plot(df_test_sc["cum_pct"], cap, label="Model")
        plt.plot(df_test_sc["cum_pct"], ideal[:len(df_test_sc)], label="İdeal", ls="--")
        plt.plot(df_test_sc["cum_pct"], random_line, label="Rastgele", ls=":")
        plt.xlabel("Kümülatif Nüfus"); plt.ylabel("Kümülatif Olay"); plt.title("CAP (Test)"); plt.legend()
        plt.tight_layout(); plt.savefig(Path(REP)/"cap_curve.png", dpi=150); plt.close()
    except Exception:
        pass

    # Scenario waterfall (PD/LGD/EAD)
    scenarios = cfg.stress.scenarios
    seg_series = df_te["segment"].values
    ead_mean_by_seg = df_te.groupby("segment")["ead0"].mean().to_dict()
    lgd_mean = float(np.nanmean(LGD_PATH[:, :12])) if not np.isnan(LGD_PATH[:, :12]).all() else 0.45
    disc12 = df12_disc
    def el12(pd_vec, lgd, ead_by_seg, seg_series):
        el = 0.0
        for s in cfg.seg:
            m = (seg_series==s)
            if m.sum()==0: continue
            el += float(pd_vec[m].mean()*lgd*ead_by_seg[s]*disc12*m.sum())
        return el
    b_g,b_u,b_h,b_pr = scenarios["Baseline"]
    pd_base = scenario_uplift(pd_use, b_g,b_u,b_h,b_pr, cfg.macro.mu)
    EL_base = el12(pd_base, lgd_mean, ead_mean_by_seg, seg_series)
    waterfalls = []
    for scen in ["Adverse","Severe"]:
        g,u,h,pr = scenarios[scen]
        pd_new = scenario_uplift(pd_use, g,u,h,pr, cfg.macro.mu)
        EL_pd = el12(pd_new, lgd_mean, ead_mean_by_seg, seg_series)
        lgd_new = min(0.98, lgd_mean*(1 + 0.10*(u - cfg.macro.mu[1]) - 0.08*(h - cfg.macro.mu[2])))
        EL_lgd = el12(pd_new, lgd_new, ead_mean_by_seg, seg_series)
        ead_new = {s: ead_mean_by_seg[s]*(1 + 0.05*(u - cfg.macro.mu[1]) - 0.03*(g - cfg.macro.mu[0])) for s in ead_mean_by_seg}
        EL_ead = el12(pd_new, lgd_new, ead_new, seg_series)
        waterfalls.append({"scenario": scen, "EL_baseline": EL_base, "ΔPD": EL_pd-EL_base, "ΔLGD": EL_lgd-EL_pd, "ΔEAD": EL_ead-EL_lgd, "EL_total": EL_ead})
    pd.DataFrame(waterfalls).to_csv(OUT/"scenario_waterfall.csv", index=False)

    # Vintage & Roll-rate
    df["vintage"] = (df["orig_month_back"]//6)*6
    vintage_summary = df.groupby(["segment","vintage"])["ECL_stage_EIR_t"].mean().reset_index()
    vintage_summary.to_csv(OUT/"vintage_ecl_means.csv", index=False)
    N,Hc = CHAINS.shape
    rr_rows = []
    sub = CHAINS[:,:min(Hc,13)]
    for sname in cfg.seg:
        idx = (df.segment==sname).values
        if idx.sum()==0: continue
        sub_s = sub[idx,:]
        for from_state in range(0,4):
            for to_state in range(0,6):
                trans = (sub_s[:,0]==from_state) & (sub_s[:,-1]==to_state)
                rr = float(np.mean(trans)) if trans.size>0 else np.nan
                rr_rows.append({"segment": sname, "from": STATE[from_state], "to": STATE[to_state], "rate": rr})
    pd.DataFrame(rr_rows).to_csv(OUT/"rollrate_12m_table.csv", index=False)

    # Drift alerts
    if (OUT/"feature_psi_valid_vs_test.csv").exists():
        psi_df2 = pd.read_csv(OUT/"feature_psi_valid_vs_test.csv")
    else:
        psi_df2 = pd.DataFrame([{"feature": c, "psi": feature_psi(np.array(Xva[c],float), np.array(Xte[c],float), bins=10)} for c in X.columns])
    alerts = []
    for _,r in psi_df2.iterrows():
        lvl = "green"
        if r.psi >= 0.25 and r.psi < 0.5: lvl = "amber"
        elif r.psi >= 0.5: lvl = "red"
        alerts.append({"feature": r.feature, "psi": float(r.psi), "level": lvl})
    json.dump({"psi_alerts": alerts}, open(OUT/"drift_alerts.json","w"), indent=2)

    # Save snapshots & artifacts
    df.to_csv(OUT/"snapshot_v4.csv", index=False)
    export_parquet_or_gz(pd.DataFrame(EAD_PATH), Path("artifacts")/"ead_path.parquet")
    export_parquet_or_gz(pd.DataFrame(LGD_PATH), Path("artifacts")/"lgd_path.parquet")
    export_parquet_or_gz(pd.DataFrame(CHAINS),  Path("artifacts")/"chains.parquet")

    # Report
    base_metrics = {"AUC": float(roc_auc_score(yte, p_te_seg)), "KS": KS, "Gini": Gini, "Brier": float(brier), "PR_AUC": float(pr_auc), "PSI_total": float(PSI_total)}
    html_path = write_html_report(df, base_metrics, EL_12M_by_seg_df, "reports/report.html")
    if cfg.extra.report_to_pdf: _ = export_pdf(html_path, "reports/report.pdf")

    # Scenario grid example
    seg_series_te = df.loc[Xte.index, "segment"]
    lgd_by_seg = {s: (float(np.nanmean(LGD_PATH[df.segment==s, :12])) if not np.isnan(LGD_PATH[:, :12]).all()
                      else (cfg.seg[s].lgd_unsec*0.8 + cfg.seg[s].lgd_sec*0.2)) for s in cfg.seg}
    ead_by_seg = df.loc[Xte.index].groupby("segment")["ead0"].mean().to_dict()
    grid = [
        cfg.stress.scenarios["Baseline"],
        cfg.stress.scenarios["Adverse"],
        cfg.stress.scenarios["Severe"],
        ( 0.01, 0.12, 0.00, 0.40),  # MildPlus
        (-0.03, 0.16,-0.03, 0.45),  # Deep
    ]
    names = ["Baseline","Adverse","Severe","MildPlus","Deep"]
    scen_grid_df = run_scenario_grid(p_te_seg, seg_series_te.values, cfg, names, grid, lgd_by_seg, ead_by_seg)
    scen_grid_df.pivot(index="segment", columns="scenario", values="EL12").fillna(0.0).to_csv(OUT/"scenario_grid_el12.csv")

    # Dashboard
    RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    dashboard = {
        "run_id": RUN_ID, "utc": datetime.utcnow().isoformat(timespec="seconds"),
        "rows": int(df.shape[0]),
        "AUC_test": float(roc_auc_score(yte, p_te_seg)),
        "ECL12_sum_t": float(df["ECL_12m_EIR_t"].sum()),
        "ECLlife_sum_t": float(df["ECL_life_EIR_t"].sum()),
        "Stage1_share": float((df["stage_ifrs9"]=="Stage1").mean()),
        "Stage2_share": float((df["stage_ifrs9"]=="Stage2").mean()),
        "Stage3_share": float((df["stage_ifrs9"]=="Stage3").mean()),
        "VaR95": VaR95, "ES95": ES95, "VaR99": VaR99, "ES99": ES99
    }
    with open(OUT/f"dashboard_{RUN_ID}.json","w") as f: json.dump(dashboard,f,indent=2)

    return {
        "df": df, "MACRO": MACRO, "CHAINS": CHAINS, "EAD_PATH": EAD_PATH, "LGD_PATH": LGD_PATH, "REC_PATH": REC_PATH,
        "metrics": base_metrics, "EL12_by_seg": EL_12M_by_seg_df, "dashboard": dashboard
    }

# ===================== CLI =====================

def load_csv(path: Optional[str]):
    if path is None: return None
    assert os.path.exists(path), f"File not found: {path}"
    return pd.read_csv(path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=str, default=None, help="Portfolio CSV (segment,income,score,ltv0,limit,util0,ead0?,tenor_m?,secured?,collateral_type?)")
    ap.add_argument("--macro", type=str, default=None, help="Macro CSV (gdp,unemp,hpi,policy)")
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config override")
    ap.add_argument("--nu", type=int, default=5, help="t-copula dof (nu)")
    ap.add_argument("--stage3", type=str, default="stop", choices=["stop","net"], help="Stage 3 interest policy")
    ap.add_argument("--pdf", action="store_true", help="Export HTML report to PDF (requires wkhtmltopdf)")
    ap.add_argument("--mc", type=int, default=200, help="Monte Carlo run count")
    return ap.parse_args()

def apply_config_overrides(cfg: SandboxCfg, path: Optional[str]):
    if not path: return cfg
    if not os.path.exists(path): return cfg
    try:
        if path.endswith((".yaml",".yml")) and HAVE_YAML:
            ov = yaml.safe_load(open(path,"r",encoding="utf-8"))
        else:
            ov = json.load(open(path,"r",encoding="utf-8"))
        if "model" in ov and "horizon_m" in ov["model"]:
            cfg.model.horizon_m = int(ov["model"]["horizon_m"])
        if "sicr" in ov:
            if "abs_threshold" in ov["sicr"]: cfg.sicr.abs_threshold = float(ov["sicr"]["abs_threshold"])
            if "rel_multiplier" in ov["sicr"]: cfg.sicr.rel_multiplier = float(ov["sicr"]["rel_multiplier"])
    except Exception:
        pass
    return cfg

if __name__ == "__main__":
    args = parse_args()
    cfg = default_config()
    cfg = apply_config_overrides(cfg, args.config)
    cfg.extra.t_copula_nu = int(args.nu)
    cfg.extra.stage3_interest_policy = args.stage3
    cfg.extra.report_to_pdf = bool(args.pdf)
    cfg.extra.mc_runs = int(args.mc)

    user_port = load_csv(args.portfolio)
    user_macro= load_csv(args.macro)
    out = run_pipeline_full(cfg, user_port, user_macro)

    print(json.dumps({
        "rows": int(out["df"].shape[0]),
        "AUC": out["metrics"]["AUC"],
        "KS": out["metrics"]["KS"],
        "Gini": out["metrics"]["Gini"],
        "Brier": out["metrics"]["Brier"],
        "PR_AUC": out["metrics"]["PR_AUC"],
        "ECL12_sum": float(out["df"]["ECL_12m_EIR_t"].sum()),
        "ECLlife_sum": float(out["df"]["ECL_life_EIR_t"].sum()),
        "report_html": "reports/report.html"
    }, indent=2))
