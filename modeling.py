# -*- coding: utf-8 -*-
# modeling.py  (GPU-aware, end-to-end)
# -------------------------------------------------------------------
# IFRS9 Sandbox – Advanced Modeling Pipeline (GPU optimized)
# - Loads panel (artifacts/ifrs9_panel.parquet)
# - Engineers snapshot features from panel
# - Trains many PD models (Logit, RF, LightGBM, XGBoost, CatBoost, MLP)
#     * Uses GPU for XGBoost / CatBoost / LightGBM when available
# - Picks best PD by AUC, applies Platt calibration
# - Builds risk score (0–100) + A–E risk bands
# - Trains LGD & EAD regressors (LightGBM GPU if available, else RF)
# - Saves models, metrics and scored snapshot to artifacts/
# -------------------------------------------------------------------

import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
import joblib

# Optional libs
try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from catboost import CatBoostClassifier
    CAT_OK = True
except Exception:
    CAT_OK = False

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

import matplotlib.pyplot as plt
import logging

# -----------------------
# IO paths
# -----------------------
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)
VAL_DIR = ART / "validation"; VAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ART / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)
PANEL_PATH = ART / "ifrs9_panel.parquet"

# -----------------------
# Utils
# -----------------------
def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

def cap_ar(y_true, y_prob):
    n = len(y_true)
    n_pos = max(1, int(np.sum(y_true)))
    idx = np.argsort(-y_prob)
    y_sorted = y_true[idx]
    cum_pos = np.cumsum(y_sorted)
    perc_obs = np.arange(1, n+1) / n
    perc_pos = cum_pos / n_pos
    auc_model = np.trapz(perc_pos, perc_obs)
    x_perf = np.array([0, n_pos/n, 1]); y_perf = np.array([0, 1, 1])
    auc_perf = np.trapz(y_perf, x_perf)
    return float((auc_model - 0.5) / (auc_perf - 0.5 + 1e-12))

def assign_risk_score_bands(pd_values: np.ndarray):
    scores = (1 - np.clip(pd_values, 0, 1)) * 100.0
    scores = np.clip(scores, 0, 100)
    bands = pd.cut(
        pd_values,
        bins=[0, 0.02, 0.05, 0.10, 0.20, 1.0],
        labels=["A - Çok Düşük Risk","B - Düşük Risk","C - Orta Risk","D - Yüksek Risk","E - Çok Yüksek Risk"]
    )
    return scores, bands

def plot_and_save_calibration(y_true, y_prob, path_png: Path, title="Calibration"):
    try:
        import matplotlib.pyplot as plt
        pt, pp = calibration_curve(y_true, y_prob, n_bins=10)
        plt.figure(figsize=(5,5))
        plt.plot(pp, pt, "o-", label="Observed")
        plt.plot([0,1],[0,1],"--", label="Ideal")
        plt.xlabel("Predicted PD"); plt.ylabel("Observed DR"); plt.title(title); plt.legend()
        plt.tight_layout(); plt.savefig(path_png, dpi=140); plt.close()
    except Exception:
        pass

def plot_and_save_roc(y_true, y_prob, path_png: Path, title="ROC"):
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'--',label="Random")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend()
        plt.tight_layout(); plt.savefig(path_png, dpi=140); plt.close()
    except Exception:
        pass

def plot_and_save_pr(y_true, y_prob, path_png: Path, title="PR"):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(5,5))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
        plt.tight_layout(); plt.savefig(path_png, dpi=140); plt.close()
    except Exception:
        pass

# -----------------------
# Load & engineer features from panel
# -----------------------
def load_panel(panel_path=PANEL_PATH):
    if not Path(panel_path).exists():
        raise FileNotFoundError(f"Panel not found: {panel_path}")
    return pd.read_parquet(panel_path)

def engineer_snapshot_features_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel'den snapshot özelliklerini çıkarır (güncellenmiş sütun adları)"""
    sev_map = {"current":0,"30dpd":1,"60dpd":2,"90dpd":3,"default":4,"closed":0}
    panel = panel.copy()
    panel["sev"] = panel["state"].map(sev_map).fillna(0).astype(int)
    
    # Lifetime PD hesaplaması (panel'de yoksa)
    if "lifetime_pd" not in panel.columns:
        panel["lifetime_pd"] = panel["pd12"] * 1.5  # Basit lifetime PD tahmini
    
    g = panel.groupby("account_id", as_index=False)
    last = g.tail(1)
    first = g.head(1)

    agg = panel.groupby("account_id").agg(
        months_observed=("month","max"),
        ead_last=("ead","last"),
        ead_mean=("ead","mean"),
        ead_std=("ead","std"),
        ead_min=("ead","min"),
        ead_max=("ead","max"),
        pd12_last=("pd12","last"),
        pd12_mean=("pd12","mean"),
        delinquent_months=("days_past_due", lambda x: (x > 0).sum()),
        any30=("days_past_due", lambda x: (x >= 30).any()),
        any60=("days_past_due", lambda x: (x >= 60).any()),
        any90=("days_past_due", lambda x: (x >= 90).any()),
        any_default=("state", lambda x: (x == 'default').any()),
        any_closed=("state", lambda x: False),  # closed yok
    ).round(6).reset_index()

    first_ead = first.set_index("account_id")["ead"]
    last_ead  = last.set_index("account_id")["ead"]
    agg["ead_first"] = agg["account_id"].map(first_ead)
    agg["ead_last_chk"] = agg["account_id"].map(last_ead)
    agg["ead_pct_change"] = (agg["ead_last_chk"] - agg["ead_first"]) / (agg["ead_first"] + 1e-9)
    agg.drop(columns=["ead_last_chk"], inplace=True)

    agg["target_default"] = agg["any_default"].astype(int)
    
    # Calculate stage_last based on delinquency and defaults
    # Stage 1: Normal, Stage 2: SICR (30-89 DPD), Stage 3: Default
    agg["stage_last"] = 1  # Default to Stage 1
    agg.loc[agg["any30"] & ~agg["any90"], "stage_last"] = 2  # Stage 2 for 30-89 DPD
    agg.loc[agg["any_default"], "stage_last"] = 3  # Stage 3 for defaults
    
    # Add lifetime_pd calculation
    agg["lifetime_pd_last"] = agg["pd12_last"] * 1.2  # Simple approximation
    agg["lifetime_pd_mean"] = agg["pd12_mean"] * 1.2  # Simple approximation
    
    for c in ["ead_std","ead_min","ead_max"]:
        agg[c] = agg[c].fillna(0.0)

    feats = [
        "months_observed","ead_last","ead_mean","ead_std","ead_min","ead_max","ead_first","ead_pct_change",
        "pd12_last","pd12_mean","lifetime_pd_last","lifetime_pd_mean",
        "delinquent_months","any30","any60","any90","any_closed"
    ]
    agg[feats] = agg[feats].astype(float)
    return agg

# -----------------------
# PD Modeling (GPU-aware)
# -----------------------
def train_many_pd_models(df_snap: pd.DataFrame, random_state=42):
    features = [
        "months_observed","ead_last","ead_mean","ead_std","ead_min","ead_max","ead_first","ead_pct_change",
        "pd12_last","pd12_mean","lifetime_pd_last","lifetime_pd_mean",
        "delinquent_months","any30","any60","any90","any_closed"
    ]
    X = df_snap[features].values
    y = df_snap["target_default"].values.astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=random_state, stratify=y)

    if len(np.unique(y_tr)) < 2:
        raise ValueError("Eğitim verisi en az iki farklı sınıf içermelidir. target_default sütununu ve panel verinizi kontrol edin.")

    models, metrics, prob_cache = {}, {}, {}

    # Logistic (CPU)
    logit = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    logit.fit(X_tr, y_tr)
    p = logit.predict_proba(X_te)[:,1]
    models["logit"] = logit
    metrics["logit"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
    prob_cache["logit"] = p

    # RandomForest (CPU)
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=50, n_jobs=-1,
                                random_state=random_state, class_weight="balanced_subsample")
    rf.fit(X_tr, y_tr)
    p = rf.predict_proba(X_te)[:,1]
    models["rf"] = rf
    metrics["rf"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
    prob_cache["rf"] = p

    # LightGBM (GPU if available)
    if LGB_OK:
        lgbm = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.03, num_leaves=64, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=0.4,
            random_state=random_state, n_jobs=-1
        )
        # monotone constraints on PD features
        mono = [1 if f in ("pd12_last","lifetime_pd_last","pd12_mean","lifetime_pd_mean") else 0 for f in features]
        try:
            lgbm.set_params(device_type="gpu")
        except Exception:
            pass
        lgbm.set_params(monotone_constraints=mono)
        lgbm.fit(X_tr, y_tr, eval_set=[(X_te,y_te)], eval_metric="auc")
        p = lgbm.predict_proba(X_te)[:,1]
        models["lgbm"] = lgbm
        metrics["lgbm"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
        prob_cache["lgbm"] = p

        # SHAP (subsample)
        if SHAP_OK:
            try:
                explainer = shap.TreeExplainer(lgbm)
                idx = np.random.choice(len(X_te), size=min(5000, len(X_te)), replace=False)
                shap_vals = explainer.shap_values(X_te[idx])
                import matplotlib.pyplot as plt
                vals = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals)>1 else shap_vals
                shap.summary_plot(vals, pd.DataFrame(X_te[idx], columns=features), show=False)
                plt.tight_layout(); plt.savefig(VAL_DIR/"pd_shap_summary_lgbm.png", dpi=140); plt.close()
            except Exception:
                pass

    # XGBoost (GPU if available)
    if XGB_OK:
        # GPU histogram + QuantileDMatrix (memory friendly)
        try:
            dtrain = xgb.QuantileDMatrix(X_tr, y_tr)
            dtest  = xgb.QuantileDMatrix(X_te, y_te)
            params = dict(
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", device="cuda", predictor="gpu_predictor",
                learning_rate=0.03, max_depth=8, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.6, random_state=random_state
            )
            xgbm = xgb.train(params, dtrain, num_boost_round=1200)
            p = xgbm.predict(dtest)
            models["xgb"] = xgbm
            metrics["xgb"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
            prob_cache["xgb"] = p
        except Exception:
            xgbm = xgb.XGBClassifier(
                n_estimators=1200, learning_rate=0.03, max_depth=8, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.6, random_state=random_state, objective="binary:logistic", eval_metric="auc"
            )
            try:
                xgbm.set_params(tree_method="hist", device="cuda", predictor="gpu_predictor")
            except Exception:
                pass
            xgbm.fit(X_tr, y_tr)
            p = xgbm.predict_proba(X_te)[:,1]
            models["xgb"] = xgbm
            metrics["xgb"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
            prob_cache["xgb"] = p

    # CatBoost (GPU if available)
    if CAT_OK:
        cat = CatBoostClassifier(
            iterations=1200, depth=8, learning_rate=0.03, l2_leaf_reg=6.0,
            loss_function="Logloss", eval_metric="AUC", random_seed=random_state, verbose=False
        )
        try:
            cat.set_params(task_type="GPU", devices="0")
        except Exception:
            pass
        cat.fit(X_tr, y_tr)
        p = cat.predict_proba(X_te)[:,1]
        models["cat"] = cat
        metrics["cat"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
        prob_cache["cat"] = p

    # MLP (CPU, hızlı)
    mlp = Pipeline([("scaler", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=(128,64), activation="relu",
                                          alpha=1e-4, learning_rate_init=3e-3,
                                          max_iter=300, random_state=random_state))])
    mlp.fit(X_tr, y_tr)
    p = mlp.predict_proba(X_te)[:,1]
    models["mlp"] = mlp
    metrics["mlp"] = {"auc": roc_auc_score(y_te, p), "ks": ks_statistic(y_te, p), "brier": brier_score_loss(y_te, p), "ar": cap_ar(y_te, p)}
    prob_cache["mlp"] = p

    # Pick best by AUC
    best_name = max(metrics.keys(), key=lambda k: metrics[k]["auc"])
    best_model = models[best_name]

    # Save plots for best
    plot_and_save_roc(y_te, prob_cache[best_name], VAL_DIR/"pd_roc_best.png", title=f"ROC ({best_name})")
    plot_and_save_pr(y_te, prob_cache[best_name], VAL_DIR/"pd_pr_best.png", title=f"PR ({best_name})")
    plot_and_save_calibration(y_te, prob_cache[best_name], VAL_DIR/"pd_calibration_best.png", title=f"Calibration ({best_name})")

    # Persist metrics & models
    save_json({k:{m:float(v[m]) for m in v} for k,v in metrics.items()}, VAL_DIR/"pd_metrics.json")
    # best
    joblib.dump({"model":best_model, "features":features}, MODEL_DIR/"pd_best.pkl")
    # all (best-effort)
    for name, mdl in models.items():
        try: joblib.dump({"model":mdl, "features":features}, MODEL_DIR/f"pd_{name}.pkl")
        except Exception: pass

    return best_model, features, metrics, (X_tr, X_te, y_tr, y_te)

# -----------------------
# Calibration (Platt)
# -----------------------
def fit_calibrator(best_model_pack, X_te, y_te):
    from sklearn.linear_model import LogisticRegression
    mdl = best_model_pack["model"]
    if hasattr(mdl, "predict_proba"):
        s = mdl.predict_proba(X_te)[:,1]
    else:
        s = mdl.predict(X_te)
        s = (s - s.min())/(s.max()-s.min() + 1e-9)
    lr = LogisticRegression(max_iter=500)
    lr.fit(s.reshape(-1,1), y_te)
    joblib.dump(lr, MODEL_DIR/"pd_platt_lr.pkl")
    return lr

def apply_calibration(mdl, X, platt_lr=None):
    if hasattr(mdl, "predict_proba"):
        s = mdl.predict_proba(X)[:,1]
    else:
        s = mdl.predict(X); s = (s - s.min())/(s.max()-s.min() + 1e-9)
    s = np.clip(s, 1e-6, 1-1e-6)
    if platt_lr is not None:
        s = platt_lr.predict_proba(s.reshape(-1,1))[:,1]
    return np.clip(s, 1e-6, 1-1e-6)

# -----------------------
# LGD & EAD regressors (GPU if LGB OK)
# -----------------------
def train_lgd_reg(panel: pd.DataFrame, df_snap: pd.DataFrame):
    # Create synthetic LGD data based on default information
    panel_lgd = panel.copy()
    
    # Simulate LGD based on state and days_past_due
    np.random.seed(42)
    panel_lgd["lgd_real"] = 0.0  # Default LGD for current accounts
    
    # Higher LGD for defaulted accounts
    default_mask = panel_lgd["state"] == "default"
    panel_lgd.loc[default_mask, "lgd_real"] = np.random.beta(2, 3, sum(default_mask)) * 0.8 + 0.1
    
    # Medium LGD for delinquent accounts
    delinq_mask = panel_lgd["state"].isin(["30dpd", "60dpd", "90dpd"])
    panel_lgd.loc[delinq_mask, "lgd_real"] = np.random.beta(1, 5, sum(delinq_mask)) * 0.4 + 0.05
    
    lgd_last = panel_lgd.sort_values(["account_id","month"]).groupby("account_id")["lgd_real"].last()
    df = df_snap.join(lgd_last, on="account_id", rsuffix="_real").dropna(subset=["lgd_real"])

    feats = [
        "pd12_last","pd12_mean","lifetime_pd_last","lifetime_pd_mean",
        "ead_last","ead_mean","ead_std","ead_min","ead_max","ead_first","ead_pct_change",
        "delinquent_months","any30","any60","any90"
    ]
    X = df[feats].values; y = df["lgd_real"].values

    if LGB_OK:
        mdl = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, num_leaves=64, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.3, random_state=42
        )
        try: mdl.set_params(device_type="gpu")
        except Exception: pass
        mdl.fit(X,y)
    else:
        mdl = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        mdl.fit(X,y)

    pred = mdl.predict(X)
    save_json({"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))},
              VAL_DIR/"lgd_metrics.json")
    joblib.dump({"model":mdl, "features":feats}, MODEL_DIR/"lgd_reg.pkl")
    return mdl, feats

def train_ead_reg(df_snap: pd.DataFrame):
    feats = [
        "pd12_last","pd12_mean","lifetime_pd_last","lifetime_pd_mean",
        "delinquent_months","any30","any60","any90",
        "ead_first","months_observed"
    ]
    X = df_snap[feats].values; y = df_snap["ead_last"].values

    if LGB_OK:
        mdl = lgb.LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, num_leaves=64, max_depth=8,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.3, random_state=42
        )
        try: mdl.set_params(device_type="gpu")
        except Exception: pass
        mdl.fit(X,y)
    else:
        mdl = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
        mdl.fit(X,y)

    pred = mdl.predict(X)
    save_json({"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))},
              VAL_DIR/"ead_metrics.json")
    joblib.dump({"model":mdl, "features":feats}, MODEL_DIR/"ead_reg.pkl")
    return mdl, feats

# -----------------------
# Portföy konsantrasyon analizi
# -----------------------
def portfolio_concentration_analysis(snapshot, outdir=None):
    """
    Portföyde büyük müşteri, ürün ve segment konsantrasyonunu analiz eder.
    Top 10 exposure, segment ve ürün bazlı temerrüt oranları, konsantrasyon metrikleri raporlanır.
    """
    try:
        # Top 10 exposure
        top_exposure = snapshot.nlargest(10, 'ead_last')[['account_id', 'ead_last', 'segment', 'product']]
        # Segment bazlı temerrüt oranı
        seg_default = snapshot.groupby('segment')['target_default'].mean()
        # Ürün bazlı temerrüt oranı
        prod_default = snapshot.groupby('product')['target_default'].mean()
        # Konsantrasyon metrikleri
        total_ead = snapshot['ead_last'].sum()
        top10_share = top_exposure['ead_last'].sum() / total_ead
        logging.info(f"Top 10 exposure oranı: {top10_share:.2%}")
        logging.info(f"Segment bazlı temerrüt oranları: {seg_default.to_dict()}")
        logging.info(f"Ürün bazlı temerrüt oranları: {prod_default.to_dict()}")
        # Sonuçları dosyaya kaydet
        if outdir:
            top_exposure.to_csv(f"{outdir}/validation/top10_exposure.csv", index=False)
        return {
            'top10_share': top10_share,
            'seg_default': seg_default.to_dict(),
            'prod_default': prod_default.to_dict()
        }
    except Exception as e:
        logging.error(f"Portföy konsantrasyon analizinde hata: {str(e)}")
        print(f"❌ Portföy konsantrasyon analizinde hata: {str(e)}")
        raise

# -----------------------
# Stress testing
# -----------------------
def stress_test(snapshot, macro_scenarios=None, outdir=None):
    """
    Farklı makro senaryolar altında portföy ECL ve temerrüt oranı hesaplar.
    """
    try:
        if macro_scenarios is None:
            macro_scenarios = {
                'baseline': {'gdp_growth': 0.03, 'unemployment': 0.08, 'macro_stress': 1.0},
                'mild_downturn': {'gdp_growth': 0.01, 'unemployment': 0.12, 'macro_stress': 1.5},
                'severe_downturn': {'gdp_growth': -0.02, 'unemployment': 0.18, 'macro_stress': 2.0}
            }
        results = {}
        for name, params in macro_scenarios.items():
            snap = snapshot.copy()
            # Makro stres etkisiyle PD ve LGD güncelle
            snap['pd_stress'] = snap['pd12_last'] * params['macro_stress']
            snap['lgd_stress'] = snap['lgd_last'] * params['macro_stress']
            snap['ecl_stress'] = snap['ead_last'] * snap['pd_stress'] * snap['lgd_stress']
            avg_ecl = snap['ecl_stress'].mean()
            default_rate = (snap['pd_stress'] > 0.08).mean()
            results[name] = {'avg_ecl': avg_ecl, 'default_rate': default_rate}
            logging.info(f"Stress test ({name}): ECL={avg_ecl:.2f}, Default Rate={default_rate:.2%}")
            if outdir:
                snap[['account_id','ecl_stress']].to_csv(f"{outdir}/validation/ecl_stress_{name}.csv", index=False)
        return results
    except Exception as e:
        logging.error(f"Stress test hatası: {str(e)}")
        print(f"❌ Stress test hatası: {str(e)}")
        raise

# -----------------------
# Limit optimizasyonu
# -----------------------
def limit_optimization(snapshot, outdir=None):
    """
    Her müşteri için risk skoruna göre önerilen limit hesaplar ve portföy limit dağılımını raporlar.
    """
    try:
        # Basit limit önerisi: düşük riskli müşteriye yüksek limit, yüksek riskliye düşük limit
        risk_score = snapshot['pd12_last'] * snapshot['lgd_last']
        min_limit = 5000
        max_limit = 500000
        snapshot['limit_suggestion'] = max_limit * (1 - risk_score)
        snapshot['limit_suggestion'] = snapshot['limit_suggestion'].clip(lower=min_limit, upper=max_limit)
        # Portföy limit dağılımı
        limit_dist = snapshot['limit_suggestion'].describe()
        logging.info(f"Limit önerisi dağılımı: {limit_dist.to_dict()}")
        if outdir:
            snapshot[['account_id','limit_suggestion']].to_csv(f"{outdir}/validation/limit_suggestion.csv", index=False)
        return snapshot['limit_suggestion']
    except Exception as e:
        logging.error(f"Limit optimizasyonunda hata: {str(e)}")
        print(f"❌ Limit optimizasyonunda hata: {str(e)}")
        raise

# -----------------------
# Tahsilat analizi
# -----------------------
def collection_analysis(snapshot, outdir=None):
    """
    Panelde ödeme davranışı ve tahsilat başarısını analiz eder, gecikmeli ödeme ve tahsilat oranlarını raporlar.
    """
    try:
        # Ödeme davranışı: payment_ratio, gecikmeli ödeme: days_past_due
        late_payment_rate = (snapshot['days_past_due'] > 30).mean()
        good_payment_rate = (snapshot['payment_ratio'] > 0.9).mean()
        collection_success = (snapshot['state'] != 'default').mean()
        logging.info(f"Gecikmeli ödeme oranı: {late_payment_rate:.2%}")
        logging.info(f"İyi ödeme oranı: {good_payment_rate:.2%}")
        logging.info(f"Tahsilat başarısı: {collection_success:.2%}")
        if outdir:
            with open(f"{outdir}/validation/collection_analysis.txt", "w") as f:
                f.write(f"Gecikmeli ödeme oranı: {late_payment_rate:.2%}\n")
                f.write(f"İyi ödeme oranı: {good_payment_rate:.2%}\n")
                f.write(f"Tahsilat başarısı: {collection_success:.2%}\n")
        return {
            'late_payment_rate': late_payment_rate,
            'good_payment_rate': good_payment_rate,
            'collection_success': collection_success
        }
    except Exception as e:
        logging.error(f"Tahsilat analizinde hata: {str(e)}")
        print(f"❌ Tahsilat analizinde hata: {str(e)}")
        raise

# -----------------------
# Portföy risk optimizasyonu
# -----------------------
def portfolio_risk_optimization(snapshot, outdir=None):
    """
    Portföyde risk-getiri optimizasyonu yapar, risk ve getiri dağılımını raporlar.
    """
    try:
        # Basit risk-getiri analizi: getiri = faiz * ead, risk = pd * lgd * ead
        interest_rate = 0.18
        snapshot['expected_return'] = interest_rate * snapshot['ead_last']
        snapshot['expected_loss'] = snapshot['pd12_last'] * snapshot['lgd_last'] * snapshot['ead_last']
        total_return = snapshot['expected_return'].sum()
        total_loss = snapshot['expected_loss'].sum()
        risk_return_ratio = total_return / (total_loss + 1e-6)
        logging.info(f"Toplam getiri: {total_return:.2f}, Toplam risk maliyeti: {total_loss:.2f}, Risk-getiri oranı: {risk_return_ratio:.2f}")
        if outdir:
            with open(f"{outdir}/validation/portfolio_optimization.txt", "w") as f:
                f.write(f"Toplam getiri: {total_return:.2f}\n")
                f.write(f"Toplam risk maliyeti: {total_loss:.2f}\n")
                f.write(f"Risk-getiri oranı: {risk_return_ratio:.2f}\n")
        return {
            'total_return': total_return,
            'total_loss': total_loss,
            'risk_return_ratio': risk_return_ratio
        }
    except Exception as e:
        logging.error(f"Portföy optimizasyonunda hata: {str(e)}")
        print(f"❌ Portföy optimizasyonunda hata: {str(e)}")
        raise

# -----------------------
# Senaryo analizi
# -----------------------
def scenario_analysis(snapshot, outdir=None):
    """
    Farklı ekonomik ve sektör senaryolarında portföy performansını analiz eder.
    """
    try:
        scenarios = {
            'normal': {'pd_adj': 1.0, 'lgd_adj': 1.0},
            'mild_stress': {'pd_adj': 1.3, 'lgd_adj': 1.1},
            'severe_stress': {'pd_adj': 1.7, 'lgd_adj': 1.3}
        }
        results = {}
        for name, adj in scenarios.items():
            snap = snapshot.copy()
            snap['pd_scenario'] = snap['pd12_last'] * adj['pd_adj']
            snap['lgd_scenario'] = snap['lgd_last'] * adj['lgd_adj']
            snap['ecl_scenario'] = snap['ead_last'] * snap['pd_scenario'] * snap['lgd_scenario']
            avg_ecl = snap['ecl_scenario'].mean()
            results[name] = {'avg_ecl': avg_ecl}
            logging.info(f"Senaryo ({name}) ECL: {avg_ecl:.2f}")
            if outdir:
                snap[['account_id','ecl_scenario']].to_csv(f"{outdir}/validation/ecl_scenario_{name}.csv", index=False)
        return results
    except Exception as e:
        logging.error(f"Senaryo analizinde hata: {str(e)}")
        print(f"❌ Senaryo analizinde hata: {str(e)}")
        raise

# -----------------------
# Kredi onay/red analitiği
# -----------------------
def approval_rejection_analysis(snapshot, outdir=None):
    """
    Kredi başvurularında onay/red oranı ve red nedenlerini analiz eder.
    """
    try:
        # Basit onay/red: risk skoru ve yeni başvuru
        risk_score = snapshot['pd12_last'] * snapshot['lgd_last']
        approved = (risk_score < 0.08) & (snapshot['new_application'] == 1)
        rejected = (risk_score >= 0.08) & (snapshot['new_application'] == 1)
        approval_rate = approved.mean()
        rejection_rate = rejected.mean()
        logging.info(f"Onay oranı: {approval_rate:.2%}, Red oranı: {rejection_rate:.2%}")
        if outdir:
            with open(f"{outdir}/validation/approval_rejection.txt", "w") as f:
                f.write(f"Onay oranı: {approval_rate:.2%}\nRed oranı: {rejection_rate:.2%}\n")
        return {'approval_rate': approval_rate, 'rejection_rate': rejection_rate}
    except Exception as e:
        logging.error(f"Onay/red analizinde hata: {str(e)}")
        print(f"❌ Onay/red analizinde hata: {str(e)}")
        raise

# -----------------------
# Vade/faiz analizi
# -----------------------
def maturity_interest_analysis(snapshot, outdir=None):
    """
    Vade ve faiz değişiminin risk ve getiri üzerindeki etkisini analiz eder.
    """
    try:
        # Vade ve faiz varyasyonları
        maturities = [12, 24, 36, 60]
        interest_rates = [0.12, 0.15, 0.18, 0.22]
        results = {}
        for m in maturities:
            for r in interest_rates:
                snap = snapshot.copy()
                snap['expected_return'] = r * snap['ead_last']
                snap['expected_loss'] = snap['pd12_last'] * snap['lgd_last'] * snap['ead_last']
                net_profit = snap['expected_return'].sum() - snap['expected_loss'].sum()
                results[f"vade_{m}_faiz_{r}"] = {'net_profit': net_profit}
                logging.info(f"Vade {m} ay, Faiz {r:.2f}: Net kâr {net_profit:.2f}")
        if outdir:
            import json
            with open(f"{outdir}/validation/maturity_interest_analysis.json", "w") as f:
                json.dump(results, f, indent=2)
        return results
    except Exception as e:
        logging.error(f"Vade/faiz analizinde hata: {str(e)}")
        print(f"❌ Vade/faiz analizinde hata: {str(e)}")
        raise

# -----------------------
# Teminat analizi
# -----------------------
def collateral_analysis(snapshot, outdir=None):
    """
    Teminat türü ve değerinin LGD ve ECL üzerindeki etkisini analiz eder.
    """
    try:
        # Teminat türü: mortgage, cash, none
        if 'collateral_type' not in snapshot.columns:
            snapshot['collateral_type'] = snapshot['product'].map({
                'Mortgage': 'mortgage',
                'Business Loan': 'cash',
                'Consumer Loan': 'none',
                'Credit Card': 'none'
            })
        collateral_adj = {'mortgage': 0.5, 'cash': 0.7, 'none': 1.0}
        snapshot['lgd_collateral'] = snapshot['lgd_last'] * snapshot['collateral_type'].map(collateral_adj)
        snapshot['ecl_collateral'] = snapshot['ead_last'] * snapshot['pd12_last'] * snapshot['lgd_collateral']
        avg_ecl = snapshot['ecl_collateral'].mean()
        logging.info(f"Teminat etkili ECL: {avg_ecl:.2f}")
        if outdir:
            snapshot[['account_id','ecl_collateral']].to_csv(f"{outdir}/validation/ecl_collateral.csv", index=False)
        return snapshot['ecl_collateral']
    except Exception as e:
        logging.error(f"Teminat analizinde hata: {str(e)}")
        print(f"❌ Teminat analizinde hata: {str(e)}")
        raise

# -----------------------
# Kârlılık analizi
# -----------------------
def profitability_analysis(snapshot, outdir=None):
    """
    Portföyün net faiz geliri, risk maliyeti ve kârlılığını analiz eder.
    """
    try:
        interest_rate = 0.18
        snapshot['net_interest_income'] = interest_rate * snapshot['ead_last']
        snapshot['risk_cost'] = snapshot['pd12_last'] * snapshot['lgd_last'] * snapshot['ead_last']
        snapshot['profit'] = snapshot['net_interest_income'] - snapshot['risk_cost']
        total_profit = snapshot['profit'].sum()
        logging.info(f"Portföy net kârı: {total_profit:.2f}")
        if outdir:
            with open(f"{outdir}/validation/profitability_analysis.txt", "w") as f:
                f.write(f"Portföy net kârı: {total_profit:.2f}\n")
        return total_profit
    except Exception as e:
        logging.error(f"Kârlılık analizinde hata: {str(e)}")
        print(f"❌ Kârlılık analizinde hata: {str(e)}")
        raise

# -----------------------
# Main wrapper for pipeline
# -----------------------
def run_models(snapshot, outdir=None):
    """
    Snapshot verisi üzerinden PD/LGD/EAD modellerini eğitir, skorlar ve validasyon metriklerini kaydeder.
    Ayrıca SHAP ile açıklanabilirlik analizi yapar.
    """
    try:
        # 1) Load panel and engineer features
        panel = load_panel(outdir / "ifrs9_panel.parquet")
        snap = engineer_snapshot_features_from_panel(panel)
        
        # 2) PD models (GPU-aware)
        best_pd_model, pd_features, pd_metrics, splits = train_many_pd_models(snap, random_state=42)
        X_tr, X_te, y_tr, y_te = splits
        
        # 3) Platt calibrator
        platt = fit_calibrator({"model":best_pd_model, "features":pd_features}, X_te, y_te)
        
        # 4) Score full population (calibrated PD)
        X_full = snap[pd_features].values
        pd_cal = apply_calibration(best_pd_model, X_full, platt_lr=platt)
        snap["pd_calibrated"] = pd_cal
        
        # 5) Risk score + bands
        snap["risk_score"], snap["risk_band"] = assign_risk_score_bands(snap["pd_calibrated"].values)
        
        # 6) LGD & EAD regressors
        lgd_model, lgd_feats = train_lgd_reg(panel, snap)
        ead_model, ead_feats = train_ead_reg(snap)
        
        # 7) Return scored snapshot and models
        models = {
            "pd_best": best_pd_model,
            "pd_features": pd_features,
            "platt": platt,
            "lgd": lgd_model,
            "ead": ead_model
        }
        
        # Model eğitimi (örnek: LightGBM)
        import lightgbm as lgb
        X = snapshot.drop(['account_id', 'target_default', 'stage_last'], axis=1).copy()
        y = snapshot['target_default']
        # Convert object columns to categorical for LightGBM
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        # Skorlar
        snapshot['pd_score'] = model.predict_proba(X)[:,1]
        # Validasyon metrikleri
        from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
        roc_auc = roc_auc_score(y, snapshot['pd_score'])
        brier = brier_score_loss(y, snapshot['pd_score'])
        logging.info(f"ROC AUC: {roc_auc:.4f}, Brier: {brier:.4f}")
        # SHAP açıklanabilirlik
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(f"{outdir}/validation/pd_shap_summary_lgbm.png")
        logging.info("SHAP summary plot kaydedildi.")
        
        return snapshot, model
    except Exception as e:
        logging.error(f"Modelleme hatası: {str(e)}")
        print(f"❌ Modelleme hatası: {str(e)}")
        raise

# -----------------------
# PD model karşılaştırma
# -----------------------
def compare_pd_models(snapshot, outdir=None):
    """
    Farklı model türleriyle PD model performansını karşılaştırır.
    Logistic Regression, Random Forest, XGBoost, LightGBM ile ROC AUC ve Brier skorları raporlanır.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score, brier_score_loss
        X = snapshot.drop(['account_id', 'target_default', 'stage_last'], axis=1)
        y = snapshot['target_default']
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'RandomForest': RandomForestClassifier(n_estimators=100),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier()
        }
        results = {}
        for name, model in models.items():
            model.fit(X, y)
            score = model.predict_proba(X)[:,1]
            roc_auc = roc_auc_score(y, score)
            brier = brier_score_loss(y, score)
            results[name] = {'roc_auc': roc_auc, 'brier': brier}
            logging.info(f"{name} ROC AUC: {roc_auc:.4f}, Brier: {brier:.4f}")
        if outdir:
            import json
            with open(f"{outdir}/validation/pd_model_comparison.json", "w") as f:
                json.dump(results, f, indent=2)
        return results
    except Exception as e:
        logging.error(f"PD model karşılaştırmada hata: {str(e)}")
        print(f"❌ PD model karşılaştırmada hata: {str(e)}")
        raise

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # 1) Load panel & engineer snapshot
    panel = load_panel(PANEL_PATH)
    snap = engineer_snapshot_features_from_panel(panel)

    # 2) PD models (GPU-aware)
    best_pd_model, pd_features, pd_metrics, splits = train_many_pd_models(snap, random_state=42)
    X_tr, X_te, y_tr, y_te = splits

    # 3) Platt calibrator
    platt = fit_calibrator({"model":best_pd_model, "features":pd_features}, X_te, y_te)

    # 4) Score full population (calibrated PD)
    X_full = snap[pd_features].values
    pd_cal = apply_calibration(best_pd_model, X_full, platt_lr=platt)
    snap["pd_calibrated"] = pd_cal

    # 5) Risk score + bands
    snap["risk_score"], snap["risk_band"] = assign_risk_score_bands(snap["pd_calibrated"].values)

    # 6) LGD & EAD regressors
    lgd_model, lgd_feats = train_lgd_reg(panel, snap)
    ead_model, ead_feats = train_ead_reg(snap)

    # 7) Persist scored snapshot and preview
    out_path = ART/"snapshot_scored.parquet"
    snap.to_parquet(out_path, index=False)
    preview_cols = ["account_id","pd12_last","lifetime_pd_last","pd_calibrated","risk_score","risk_band",
                    "ead_last","delinquent_months","any90","stage_last","target_default"]
    snap.head(5000)[[c for c in preview_cols if c in snap.columns]].to_csv(ART/"snapshot_scored_preview.csv", index=False, encoding="utf-8")

    # 8) Save calibrated plots
    plot_and_save_roc(y_te, apply_calibration(best_pd_model, X_te, platt), VAL_DIR/"pd_roc_calibrated.png", title="ROC (Calibrated)")
    plot_and_save_pr(y_te, apply_calibration(best_pd_model, X_te, platt), VAL_DIR/"pd_pr_calibrated.png", title="PR (Calibrated)")
    plot_and_save_calibration(y_te, apply_calibration(best_pd_model, X_te, platt), VAL_DIR/"pd_calibration_calibrated.png", title="Calibration (Calibrated)")

    # 9) Metadata
    meta = {
        "pd_best_model": "pd_best.pkl",
        "pd_platt_lr": "pd_platt_lr.pkl",
        "lgd_model": "lgd_reg.pkl",
        "ead_model": "ead_reg.pkl",
        "features_pd": pd_features,
        "features_lgd": lgd_feats,
        "features_ead": ead_feats,
        "notes": "GPU enabled for XGBoost/CatBoost/LightGBM when available. Risk score = (1 - PD) * 100."
    }
    save_json(meta, ART/"modeling_meta.json")
    print("[OK] modeling.py completed. Saved to artifacts/")
def assign_credit_rating(snapshot, outdir=None):
    """
    Müşteri risk skoruna göre içsel kredi ratingi atar ve portföy rating dağılımını raporlar.
    """
    try:
        # Basit rating: AAA, AA, A, BBB, BB, B, CCC
        risk_score = snapshot['pd12_last'] * snapshot['lgd_last']
        bins = [0, 0.02, 0.04, 0.07, 0.10, 0.15, 0.25, 1.0]
        labels = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        snapshot['credit_rating'] = pd.cut(risk_score, bins=bins, labels=labels, right=False)
        rating_dist = snapshot['credit_rating'].value_counts().sort_index()
        logging.info(f"Portföy rating dağılımı: {rating_dist.to_dict()}")
        if outdir:
            rating_dist.to_csv(f"{outdir}/validation/credit_rating_distribution.csv")
        return snapshot['credit_rating']
    except Exception as e:
        logging.error(f"Kredi rating atamasında hata: {str(e)}")
        print(f"❌ Kredi rating atamasında hata: {str(e)}")
        raise
