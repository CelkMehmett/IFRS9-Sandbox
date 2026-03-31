# -*- coding: utf-8 -*-
# report.py
# -------------------------------------------------------------------
# IFRS9 Sandbox - Auto HTML/PDF Report Generator
# - Reads outputs from:
#     artifacts/snapshot_scored.parquet
#     artifacts/ecl/account_ecl.parquet
#     artifacts/ecl/summary_by_segment.csv
#     artifacts/ecl/summary_by_stage.csv
#     artifacts/ecl/summary_by_scenario.csv
#     artifacts/ecl/portfolio_summary.json
#     artifacts/validation/pd_metrics.json
#     artifacts/validation/*.png (optional plots)
# - Builds a single self-contained HTML report (embeds images as base64)
# - Optional PDF export via pdfkit / weasyprint if available (best-effort)
# - Saves under artifacts/report/report.html (and report.pdf if possible)
# -------------------------------------------------------------------

import os
import io
import json
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = Path("artifacts")
VAL_DIR = ART / "validation"
ECL_DIR = ART / "ecl"
REP_DIR = ART / "report"
REP_DIR.mkdir(parents=True, exist_ok=True)

SNAP_SCORED = ART / "snapshot_scored.parquet"
ACC_ECL     = ECL_DIR / "account_ecl.parquet"
SEG_SUM     = ECL_DIR / "summary_by_segment.csv"
STG_SUM     = ECL_DIR / "summary_by_stage.csv"
SCN_SUM     = ECL_DIR / "summary_by_scenario.csv"
PORT_SUM    = ECL_DIR / "portfolio_summary.json"
PD_METRICS  = VAL_DIR / "pd_metrics.json"
LTV_SUM     = ECL_DIR / "summary_by_ltv_band.csv"

IMG_FILES = {
    "roc": VAL_DIR / "pd_roc_best.png",
    "pr": VAL_DIR / "pd_pr_best.png",
    "calib": VAL_DIR / "pd_calibration_best.png",
    "roc_cal": VAL_DIR / "pd_roc_calibrated.png",
    "pr_cal": VAL_DIR / "pd_pr_calibrated.png",
    "calib_cal": VAL_DIR / "pd_calibration_calibrated.png",
    "shap": VAL_DIR / "pd_shap_summary_lgbm.png",
}

# -----------------------------
# Utils
# -----------------------------
def fmt_money(x):
    try:
        return f"{float(x):,.0f}".replace(",", " ")
    except Exception:
        return str(x)

def fmt_pct(x, digits=2):
    try:
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return "-"

def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # infer mime
    ext = path.suffix.lower().replace(".", "")
    mime = "png" if ext in ("png", "apng") else ("jpeg" if ext in ("jpg","jpeg") else "png")
    return f"data:image/{mime};base64,{b64}"

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def safe_read_parquet(path: Path, fallback=None):
    if path.exists():
        return pd.read_parquet(path)
    return fallback

def safe_read_csv(path: Path, fallback=None):
    if path.exists():
        return pd.read_csv(path)
    return fallback

def safe_read_json(path: Path, fallback=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return fallback

# -----------------------------
# Derived plots (created here)
# -----------------------------
def plot_risk_score_hist(snap: pd.DataFrame) -> str:
    if "risk_score" not in snap.columns:
        return ""
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(snap["risk_score"].astype(float), bins=30)
    ax.set_title("Risk Skoru Dağılımı (0-100)")
    ax.set_xlabel("Risk Skoru")
    ax.set_ylabel("Adet")
    return fig_to_base64(fig)

def plot_pd_by_band(snap: pd.DataFrame) -> str:
    if "risk_band" not in snap.columns or "pd_calibrated" not in snap.columns:
        return ""
    grp = snap.groupby("risk_band")["pd_calibrated"].mean().reset_index()
    order = ["A - Çok Düşük Risk","B - Düşük Risk","C - Orta Risk","D - Yüksek Risk","E - Çok Yüksek Risk"]
    grp["risk_band"] = pd.Categorical(grp["risk_band"], categories=order, ordered=True)
    grp = grp.sort_values("risk_band")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(grp["risk_band"].astype(str), grp["pd_calibrated"].values)
    ax.set_title("Risk Bandına Göre Ortalama PD")
    ax.set_xlabel("Risk Bandı")
    ax.set_ylabel("Ortalama PD")
    plt.xticks(rotation=20)
    return fig_to_base64(fig)

def plot_segment_coverage(seg_sum: pd.DataFrame) -> str:
    if seg_sum is None or seg_sum.empty:
        return ""
    fig, ax = plt.subplots(figsize=(7,4))
    tmp = seg_sum.copy()
    tmp["segment"] = tmp["segment"].astype(str)
    ax.bar(tmp["segment"], tmp["covlife"].astype(float))
    ax.set_title("Segment Bazlı Lifetime ECL Coverage")
    ax.set_xlabel("Segment")
    ax.set_ylabel("ECL / EAD")
    plt.xticks(rotation=30)
    return fig_to_base64(fig)

def plot_ltv_coverage(ltv_sum: pd.DataFrame) -> str:
    if ltv_sum is None or ltv_sum.empty:
        return ""
    fig, ax = plt.subplots(figsize=(7,4))
    tmp = ltv_sum.copy()
    tmp["ltv_band"] = tmp["ltv_band"].astype(str)
    ax.plot(tmp["ltv_band"], tmp["covlife"].astype(float), marker='o')
    ax.set_title("LTV Bandına Göre Lifetime ECL Coverage")
    ax.set_xlabel("LTV Bandı")
    ax.set_ylabel("ECL / EAD")
    return fig_to_base64(fig)

def top_accounts_table(acc_ecl: pd.DataFrame, topn=20):
    if acc_ecl is None or acc_ecl.empty:
        return pd.DataFrame()
    df = acc_ecl.copy()
    # Fallbacks: stage_calc yerine stage; pd_calibrated yerine pd12_last veya pd_score
    stage_col = "stage_calc" if "stage_calc" in df.columns else ("stage" if "stage" in df.columns else None)
    pd_col = None
    for c in ["pd_calibrated", "pd12_last", "pd_score"]:
        if c in df.columns:
            pd_col = c
            break
    cols = [c for c in ["account_id", "segment", stage_col, pd_col, "ead_last", "ecl_lifetime_weighted"] if c and c in df.columns]
    if not cols:
        return pd.DataFrame()
    df = df[cols].copy()
    # Normalize column names for display consistency
    if stage_col and stage_col != "stage_calc":
        df = df.rename(columns={stage_col: "stage_calc"})
    if pd_col and pd_col != "pd_calibrated":
        df = df.rename(columns={pd_col: "pd_calibrated"})
    sort_col = "ecl_lifetime_weighted" if "ecl_lifetime_weighted" in df.columns else ("ecl12_weighted" if "ecl12_weighted" in df.columns else None)
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    return df.head(topn)

# -----------------------------
# HTML Render
# -----------------------------
def render_html(context: dict) -> str:
    css = """
    <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #1f2937; }
    h1 { font-size: 28px; margin-bottom: 4px; }
    h2 { font-size: 22px; margin-top: 28px; }
    h3 { font-size: 18px; margin-top: 18px; }
    .muted { color: #6b7280; }
    .kpi { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
    .big { font-size: 22px; font-weight: 700; }
    .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .img { border: 1px solid #e5e7eb; border-radius: 12px; background: #fff; padding: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border-bottom: 1px solid #f3f4f6; padding: 8px; font-size: 13px; text-align: right; }
    th { text-align: left; background: #f9fafb; }
    .foot { margin-top: 32px; font-size: 12px; color: #6b7280; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 12px; background: #eef2ff; color: #3730a3; }
    </style>
    """
    # Build tables HTML
    def df_to_html(df: pd.DataFrame, money_cols=None, pct_cols=None):
        if df is None or df.empty:
            return "<p class='muted'>Veri yok.</p>"
        df = df.copy()
        money_cols = set(money_cols or [])
        pct_cols = set(pct_cols or [])
        # format
        for c in df.columns:
            if c in money_cols:
                df[c] = df[c].apply(fmt_money)
            if c in pct_cols:
                df[c] = (df[c].astype(float)*100).map(lambda v: f"{v:.2f}%")
        # render
        cols = "".join([f"<th>{c}</th>" for c in df.columns])
        rows = []
        for _, r in df.iterrows():
            cells = "".join([f"<td>{r[c]}</td>" for c in df.columns])
            rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{cols}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"IFRS 9 Kredi Riski Raporu — {now}"

    # KPIs
    kpi_html = f"""
    <div class="kpi">
      <div class="card"><div class="label">Toplam EAD</div><div class="big">{fmt_money(context['kpis'].get('total_ead', 0))}</div></div>
      <div class="card"><div class="label">12A ECL (Ağırlıklı)</div><div class="big">{fmt_money(context['kpis'].get('total_ecl12_w', 0))}</div></div>
      <div class="card"><div class="label">Lifetime ECL (Ağırlıklı)</div><div class="big">{fmt_money(context['kpis'].get('total_ecllife_w', 0))}</div></div>
      <div class="card"><div class="label">Coverage 12A</div><div class="big">{context['kpis'].get('coverage_12m_str', '0.00%')}</div></div>
      <div class="card"><div class="label">Coverage Lifetime</div><div class="big">{context['kpis'].get('coverage_life_str', '0.00%')}</div></div>
      <div class="card"><div class="label">En İyi PD Model</div><div class="big">{context['kpis'].get('best_model', '-')}</div></div>
    </div>
    """

    seg_html = df_to_html(context["seg_summary"], money_cols={"ead","ecl12","ecllife"}, pct_cols={"cov12","covlife"})
    stg_html = df_to_html(context["stg_summary"], money_cols={"ead","ecl12","ecllife"}, pct_cols={"cov12","covlife"})
    scn_html = df_to_html(context["scn_summary"])
    top_html = df_to_html(context["top_accounts"], money_cols={"ead_last","ecl_lifetime_weighted"})

    # Images
    img_roc = f"<img class='img' src='{context['images'].get('roc','')}' width='420'/>" if context["images"].get("roc") else ""
    img_pr = f"<img class='img' src='{context['images'].get('pr','')}' width='420'/>" if context["images"].get("pr") else ""
    img_cal = f"<img class='img' src='{context['images'].get('calib','')}' width='420'/>" if context["images"].get("calib") else ""
    img_shap = f"<img class='img' src='{context['images'].get('shap','')}' width='860'/>" if context["images"].get("shap") else ""
    img_risk_hist = f"<img class='img' src='{context['images'].get('risk_hist','')}' width='420'/>" if context["images"].get("risk_hist") else ""
    img_pd_band = f"<img class='img' src='{context['images'].get('pd_band','')}' width='420'/>" if context["images"].get("pd_band") else ""
    img_seg_cov = f"<img class='img' src='{context['images'].get('seg_cov','')}' width='860'/>" if context["images"].get("seg_cov") else ""
    img_ltv_cov = f"<img class='img' src='{context['images'].get('ltv_cov','')}' width='860'/>" if context["images"].get("ltv_cov") else ""

    # Executive summary (plain-language)
    exec_html = context.get("exec_summary_html", "")

    html = f"""
    <!doctype html>
    <html lang="tr">
    <head>
      <meta charset="utf-8" />
      <title>{title}</title>
      {css}
    </head>
    <body>
      <h1>{title}</h1>
      <div class="muted">Otomatik oluşturuldu • IFRS 9 Sandbox</div>

            {kpi_html}

            <h2>0) Yönetici Özeti (Teknik Olmayan)</h2>
            <div class="card">
                {exec_html}
            </div>

      <h2>1) Portföy Özeti</h2>
      <div class="row">
        <div class="card">
          <h3>Segment Özeti</h3>
          {seg_html}
        </div>
        <div class="card">
          <h3>Stage Özeti</h3>
          {stg_html}
        </div>
      </div>

      <div class="card">
        <h3>Senaryo Ağırlıkları & Toplam ECL</h3>
        {scn_html}
      </div>

      <h2>2) Risk Dağılımları</h2>
      <div class="row">
        <div class="card">{img_risk_hist}</div>
        <div class="card">{img_pd_band}</div>
      </div>
            <div class="row">
                <div class="card">
                    <h3>Segment Bazlı Lifetime Coverage</h3>
                    {img_seg_cov}
                </div>
                <div class="card">
                    <h3>LTV Bandına Göre Lifetime Coverage</h3>
                    {img_ltv_cov}
                </div>
            </div>

      <h2>3) Model Performansı</h2>
      <div class="row">
        <div class="card">{img_roc}</div>
        <div class="card">{img_pr}</div>
      </div>
      <div class="card">{img_cal}</div>
      <div class="card">
        <h3>SHAP Özeti (LightGBM)</h3>
        {img_shap if img_shap else "<p class='muted'>SHAP grafiği mevcut değil.</p>"}
      </div>

      <h2>4) En Yüksek EL Hesaplı İlk 20 Hesap</h2>
      <div class="card">
        {top_html}
      </div>

      <div class="foot">
        Notlar:
        <ul>
          <li>Risk Skoru = (1 - Kalibre PD) × 100. A:≤%2, B:≤%5, C:≤%10, D:≤%20, E:&gt;%20</li>
          <li>Stage 1 → 12A ECL, Stage 2/3 → Lifetime ECL kullanılır.</li>
          <li>Bu rapor sentetik verilerle üretilmiştir; üretim amaçlı değildir.</li>
        </ul>
      </div>
    </body>
    </html>
    """
    return html

# -----------------------------
# Main build
# -----------------------------
def main():
    # Load artifacts
    snap = safe_read_parquet(SNAP_SCORED, fallback=pd.DataFrame())
    acc_ecl = safe_read_parquet(ACC_ECL, fallback=pd.DataFrame())
    seg_sum = safe_read_csv(SEG_SUM, fallback=pd.DataFrame())
    stg_sum = safe_read_csv(STG_SUM, fallback=pd.DataFrame())
    scn_sum = safe_read_csv(SCN_SUM, fallback=pd.DataFrame())
    port_sum = safe_read_json(PORT_SUM, fallback={})
    ltv_sum = safe_read_csv(LTV_SUM, fallback=pd.DataFrame())
    pd_metrics = safe_read_json(PD_METRICS, fallback={})

    # Derived charts
    risk_hist_b64 = plot_risk_score_hist(snap) if not snap.empty else ""
    pd_band_b64 = plot_pd_by_band(snap) if not snap.empty else ""
    seg_cov_b64 = plot_segment_coverage(seg_sum) if seg_sum is not None and not seg_sum.empty else ""
    ltv_cov_b64 = plot_ltv_coverage(ltv_sum) if ltv_sum is not None and not ltv_sum.empty else ""

    # Base images
    images = {k: img_to_base64(v) for k, v in IMG_FILES.items()}
    # add derived
    images["risk_hist"] = risk_hist_b64
    images["pd_band"]   = pd_band_b64
    images["seg_cov"]   = seg_cov_b64
    images["ltv_cov"]   = ltv_cov_b64

    # KPIs
    kpis = {
        "total_ead": port_sum.get("total_ead", 0.0),
        "total_ecl12_w": port_sum.get("total_ecl12_w", 0.0),
        "total_ecllife_w": port_sum.get("total_ecllife_w", 0.0),
        "coverage_12m_str": f"{(port_sum.get('coverage_12m',0.0)*100):.2f}%",
        "coverage_life_str": f"{(port_sum.get('coverage_life',0.0)*100):.2f}%",
        "best_model": max(pd_metrics.keys(), key=lambda k: pd_metrics[k]["auc"]) if pd_metrics else "-"
    }

    # Executive summary (plain-language) builder
    def make_exec_summary():
        parts = []
        # 1) Portföy boyutu (çok basit)
        total_accts = len(snap) if snap is not None else 0
        total_ead = kpis.get('total_ead', 0.0)
        parts.append(f"Portföy boyutu: <b>{total_accts:,}</b> hesap, toplam EAD <b>{fmt_money(total_ead)}</b>.")

        # 2) Beklenen zarar (yalnızca iki ana sayı)
        parts.append(f"Beklenen kredi zararı: 12 ay <b>{fmt_money(kpis['total_ecl12_w'])}</b>, toplam (lifetime) <b>{fmt_money(kpis['total_ecllife_w'])}</b>.")
        parts.append(f"Ortalama kapsama: 12 ay <b>{kpis['coverage_12m_str']}</b>, lifetime <b>{kpis['coverage_life_str']}</b>.")

        # 3) Stage dağılımı (Stage 1/2/3 toplam oranları)
        if stg_sum is not None and not stg_sum.empty and 'ead' in stg_sum.columns and stg_sum['ead'].sum() > 0:
            stg = stg_sum.copy()
            stg['share'] = stg['ead'] / stg['ead'].sum()
            s1 = stg.loc[stg.get('stage', stg.index)==1, 'share'].sum() if ('stage' in stg.columns) else None
            s2 = stg.loc[stg.get('stage', stg.index)==2, 'share'].sum() if ('stage' in stg.columns) else None
            s3 = stg.loc[stg.get('stage', stg.index)==3, 'share'].sum() if ('stage' in stg.columns) else None
            parts.append(f"Stage dağılımı: S1 {fmt_pct(s1 or 0)}, S2 {fmt_pct(s2 or 0)}, S3 {fmt_pct(s3 or 0)}.")

        # 4) En riskli segment(ler) (yalnızca ilk 2)
        if seg_sum is not None and not seg_sum.empty and 'covlife' in seg_sum.columns:
            seg = seg_sum.copy()
            seg = seg[['segment','covlife']].dropna()
            if not seg.empty:
                seg = seg.sort_values('covlife', ascending=False).head(2)
                parts.append("Öne çıkan riskli segmentler: " + ", ".join([f"{r['segment']} ({fmt_pct(float(r['covlife']))})" for _, r in seg.iterrows()]) + ".")

        # 5) Senaryo (en yüksek ağırlık)
        if scn_sum is not None and not scn_sum.empty and 'weight' in scn_sum.columns:
            top = scn_sum.sort_values('weight', ascending=False).head(1)
            if len(top) == 1:
                r = top.iloc[0]
                parts.append(f"En yüksek senaryo ağırlığı: <b>{r['scenario']}</b> ({fmt_pct(float(r['weight']))}).")

        # 6) Model etiketi (opsiyonel, tek satır)
        if kpis.get('best_model') and kpis['best_model'] != '-':
            parts.append(f"Kullanılan PD modeli: <b>{kpis['best_model']}</b>.")

        lis = "".join([f"<li>{p}</li>" for p in parts])
        return f"<ul>{lis}</ul>"

    exec_summary_html = make_exec_summary()

    # Top accounts table
    top_df = top_accounts_table(acc_ecl, topn=20)

    # Build context
    ctx = {
        "kpis": kpis,
        "seg_summary": seg_sum,
        "stg_summary": stg_sum,
        "scn_summary": scn_sum,
        "top_accounts": top_df,
    "images": images,
    "exec_summary_html": exec_summary_html
    }

    # Render HTML
    html = render_html(ctx)
    html_path = REP_DIR / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML report written: {html_path}")

    # Try PDF export (best-effort)
    pdf_path = REP_DIR / "report.pdf"
    exported = False

    # 1) pdfkit (wkhtmltopdf required)
    try:
        import pdfkit
        pdfkit.from_file(str(html_path), str(pdf_path))
        exported = pdf_path.exists()
    except Exception:
        exported = False

    # 2) weasyprint
    if not exported:
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            exported = pdf_path.exists()
        except Exception:
            exported = False

    if exported:
        print(f"[OK] PDF report written:   {pdf_path}")
    else:
        print("[INFO] PDF export not available. Install pdfkit (with wkhtmltopdf) or weasyprint to enable PDF output.")

def build_report(outdir=None, metrics=None, ecl_results=None):
    """Public API function for generating reports"""
    main()

if __name__ == "__main__":
    main()
