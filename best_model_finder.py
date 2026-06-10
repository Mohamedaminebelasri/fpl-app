"""
best_model_finder.py — V4
Walk-Forward Validation + XGBoost / LightGBM / CatBoost + Optuna + Ensemble
"""

import os, sys, warnings, subprocess, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── Auto-install ─────────────────────────────────────────────
def _install(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"Installation {pkg} ...", flush=True)
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', pkg, '--quiet'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

for _p in ['xgboost', 'lightgbm', 'catboost', 'optuna', 'tqdm']:
    _install(_p)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constantes ───────────────────────────────────────────────
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
N_TRIALS  = 30
BASELINE  = {'MAE': 1.03, 'RMSE': 2.01, 'R2': 0.31, 'Top10': 10.0}
PREV_V3   = {'MAE': 0.95, 'RMSE': 1.86, 'R2': 0.28, 'Top10': 6.0}

# Walk-forward : (train_end_gw, test_start_gw, test_end_gw)
WALK_FOLDS = [(12, 13, 16), (16, 17, 20), (20, 21, 24), (24, 25, 28), (28, 29, 32)]

t0_global = time.time()

# ============================================================
# SECTION 1 — Features V4 + cache_merged_v3.pkl
# ============================================================
print("=" * 58)
print("  SECTION 1 — FEATURES V4 (nouvelles + rolling base)")
print("=" * 58)

if os.path.exists('cache_merged_v3.pkl'):
    df = pickle.load(open('cache_merged_v3.pkl', 'rb'))
    print(f"✅ Features V4 chargées depuis cache  "
          f"({len(df):,} lignes, {len(df.columns)} col.)")
else:
    print("Chargement cache_merged_v2.pkl ...", end=' ', flush=True)
    df = pickle.load(open('cache_merged_v2.pkl', 'rb'))
    print(f"OK  ({len(df):,} lignes, {len(df.columns)} col.)")

    # Normalise AM → MID
    df['position'] = df['position'].replace('AM', 'MID')
    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

    # ── Helpers rolling ─────────────────────────────────────
    def rn_mean(col, n):
        return df.groupby(['name', 'season'])[col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean()
        )

    def rn_sum(col, n):
        return df.groupby(['name', 'season'])[col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).sum()
        )

    # ── Rolling-4 features (base repris de evaluation_model) ─
    df['form']             = rn_mean('total_points', 4)
    df['bonus_rate']       = rn_mean('bonus', 4)
    df['xg_rolling']       = rn_mean('expected_goals', 4)
    df['xa_rolling']       = rn_mean('expected_assists', 4)
    df['xgc_rolling']      = rn_mean('expected_goals_conceded', 4)
    df['saves_rolling']    = rn_mean('saves', 4)
    df['clean_sheet_rate'] = rn_mean('clean_sheets', 4)
    df['goals_rolling']    = rn_mean('goals_scored', 4)
    df['assists_rolling']  = rn_mean('assists', 4)
    df['mins_rolling']     = rn_mean('minutes', 4)

    safe_p90 = (df['mins_rolling'] / 90).clip(lower=0.1)
    df['xg_per90_r']    = (df['xg_rolling']   / safe_p90).clip(0, 5)
    df['xa_per90_r']    = (df['xa_rolling']   / safe_p90).clip(0, 3)
    df['xgc_per90_r']   = (df['xgc_rolling'] / safe_p90).clip(0, 5)
    df['saves_per90_r'] = (df['saves_rolling']/ safe_p90).clip(0, 15)

    # ── A) Nouvelles features V4 ─────────────────────────────
    df['bonus_rate_5']  = rn_mean('bps', 5)
    df['bps_rolling_3'] = rn_mean('bps', 3)
    df['minutes_pct']   = (df['minutes'] / 90).clip(0, 1)
    df['cs_rate']       = rn_mean('clean_sheets', 10)

    rm4_goals   = rn_sum('goals_scored', 4)
    rm4_assists = rn_sum('assists', 4)
    rm4_pts     = rn_sum('total_points', 4)
    rm4_mins    = rn_sum('minutes', 4).clip(lower=1)

    df['goals_per90']   = (rm4_goals   / rm4_mins * 90).clip(0, 5)
    df['assists_per90'] = (rm4_assists / rm4_mins * 90).clip(0, 4)
    df['pts_per_90']    = (rm4_pts     / rm4_mins * 90).clip(0, 15)
    df['value_form']    = (rn_mean('total_points', 4) / df['value'].clip(lower=1)).clip(0, 5)
    df['home_advantage']= df['was_home'].astype(float) * 0.3

    gc_col = 'opponent_goals_conceded_avg'
    if gc_col in df.columns:
        gc_min, gc_max = df[gc_col].min(), df[gc_col].max()
        df['opponent_strength'] = ((df[gc_col] - gc_min) / (gc_max - gc_min + 1e-8)).clip(0, 1)
    else:
        df['opponent_strength'] = 0.5

    pickle.dump(df, open('cache_merged_v3.pkl', 'wb'))
    print(f"✅ cache_merged_v3.pkl sauvegardé  ({len(df.columns)} colonnes)")

# Normalise AM → MID si chargé depuis cache
df['position'] = df['position'].replace('AM', 'MID')

# ── Target ───────────────────────────────────────────────────
df['target'] = df.groupby(['name', 'season'])['total_points'].shift(-1)
df = df.dropna(subset=['target', 'form']).reset_index(drop=True)
df['target'] = df['target'].astype(float)
print(f"Lignes prêtes : {len(df):,}\n")

# ── Feature lists par position ───────────────────────────────
NEW_V4 = [
    'bonus_rate_5', 'bps_rolling_3', 'minutes_pct',
    'goals_per90', 'assists_per90', 'cs_rate',
    'pts_per_90', 'value_form', 'home_advantage', 'opponent_strength',
]
COMMON = [
    'minutes', 'mins_rolling', 'form', 'total_points', 'bps', 'ict_index', 'value',
    'bonus_rate', 'opponent_team', 'was_home',
    'pts_rolling_3', 'pts_rolling_5', 'form_trend', 'mins_rolling_3',
    'opponent_goals_conceded_avg', 'is_top6_opponent', 'is_endseason',
] + NEW_V4

FEATURES_BY_POS = {
    'GK': COMMON + [
        'saves_rolling', 'saves_per90_r', 'clean_sheet_rate',
        'xgc_rolling', 'xgc_per90_r',
        'q_saves_per90', 'q_cs_per90', 'q_xgc_per90', 'opponent_gc_avg_def',
    ],
    'DEF': COMMON + [
        'influence', 'threat', 'clean_sheet_rate',
        'xgc_rolling', 'xgc_per90_r', 'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'q_xgc_per90', 'q_cs_per90', 'q_xg_per90', 'q_xa_per90', 'opponent_gc_avg_def',
    ],
    'MID': COMMON + [
        'influence', 'creativity', 'threat', 'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'xG_rolling_3', 'xG_rolling_5',
        'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    ],
    'FWD': COMMON + [
        'influence', 'creativity', 'threat', 'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'xG_rolling_3', 'xG_rolling_5',
        'q_xg_per90', 'q_xa_per90',
    ],
}

# ── Splits ───────────────────────────────────────────────────
# Optuna val = fold 5 (GW29-32) → pas de fuite sur test GW33+
optuna_tr_mask  = (df['season'] == '2023-24') | ((df['season'] == '2024-25') & (df['GW'] <= 28))
optuna_val_mask = (df['season'] == '2024-25') & df['GW'].between(29, 32)
final_tr_mask   = (df['season'] == '2023-24') | ((df['season'] == '2024-25') & (df['GW'] <= 32))
test_mask       = (df['season'] == '2024-25') & (df['GW'] >= 33)

df_otr  = df[optuna_tr_mask].reset_index(drop=True)
df_oval = df[optuna_val_mask].reset_index(drop=True)
df_ftr  = df[final_tr_mask].reset_index(drop=True)
df_te   = df[test_mask].reset_index(drop=True)

print(f"Optuna  train={len(df_otr):,}  val={len(df_oval):,}")
print(f"Final   train={len(df_ftr):,}  test={len(df_te):,}\n")

# ============================================================
# SECTION 2 — Walk-Forward helper
# ============================================================
def top10_accuracy(df_eval: pd.DataFrame) -> float:
    hits, n_gw = 0, 0
    for gw in sorted(df_eval['GW'].unique()):
        gw_d = df_eval[df_eval['GW'] == gw]
        if len(gw_d) < 10:
            continue
        hits += len(set(gw_d.nlargest(10, 'y_pred').index) &
                    set(gw_d.nlargest(10, 'target').index))
        n_gw += 1
    return (hits / (n_gw * 10)) * 100 if n_gw > 0 else 0.0


def walk_forward_scores(model_factory, pos: str, feats: list) -> dict:
    """Évalue un modèle sur les 5 folds walk-forward → MAE moyen."""
    fold_mae = []
    for train_end, ts, te in WALK_FOLDS:
        tr_m = (
            ((df['season'] == '2023-24') |
             ((df['season'] == '2024-25') & (df['GW'] <= train_end))) &
            (df['position'] == pos)
        )
        te_m = (
            (df['season'] == '2024-25') &
            df['GW'].between(ts, te) &
            (df['position'] == pos)
        )
        d_tr = df[tr_m];  d_te = df[te_m]
        if len(d_tr) < 30 or len(d_te) < 5:
            continue
        X_tr = d_tr[feats].fillna(0).values;  y_tr = d_tr['target'].values
        w_tr = np.where(y_tr >= 8, 1.5, 1.0)
        X_te = d_te[feats].fillna(0).values;  y_te = d_te['target'].values
        m = model_factory()
        try:
            m.fit(X_tr, y_tr, sample_weight=w_tr)
        except Exception:
            m.fit(X_tr, y_tr)
        fold_mae.append(mean_absolute_error(y_te, m.predict(X_te)))
    return {'WF_MAE': float(np.mean(fold_mae)) if fold_mae else np.nan}

# ============================================================
# SECTION 3 — Optuna : 3 modèles × 4 positions
# ============================================================
print("=" * 58)
print(f"  SECTION 3 — OPTUNA  ({N_TRIALS} trials × 3 modèles × 4 pos.)")
print("=" * 58)
print(f"  Total : {N_TRIALS * 3 * 4} trials — validation sur GW 29-32\n")

# ── Objectifs Optuna ─────────────────────────────────────────
def xgb_obj(trial, Xtr, ytr, wtr, Xv, yv):
    p = {
        'n_estimators'    : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'       : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    m = XGBRegressor(**p, n_jobs=-1, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return mean_absolute_error(yv, m.predict(Xv))

def lgb_obj(trial, Xtr, ytr, wtr, Xv, yv):
    p = {
        'n_estimators'    : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'       : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves'      : trial.suggest_int('num_leaves', 20, 100),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    m = LGBMRegressor(**p, n_jobs=-1, random_state=42, verbose=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return mean_absolute_error(yv, m.predict(Xv))

def cat_obj(trial, Xtr, ytr, wtr, Xv, yv):
    p = {
        'iterations'   : trial.suggest_int('iterations', 100, 400),
        'depth'        : trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg'  : trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    }
    m = CatBoostRegressor(**p, thread_count=-1, random_seed=42, verbose=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return mean_absolute_error(yv, m.predict(Xv))

# ── Constructeurs finaux ─────────────────────────────────────
def build_xgb(p):
    return XGBRegressor(**p, n_jobs=-1, random_state=42, verbosity=0)

def build_lgb(p):
    return LGBMRegressor(**p, n_jobs=-1, random_state=42, verbose=-1)

def build_cat(p):
    return CatBoostRegressor(**p, thread_count=-1, random_seed=42, verbose=0)

MODEL_CFG = [
    ('XGBoost',  'XGB', xgb_obj, build_xgb),
    ('LightGBM', 'LGB', lgb_obj, build_lgb),
    ('CatBoost', 'CAT', cat_obj, build_cat),
]

# ── Callback tqdm ────────────────────────────────────────────
class _TqdmCB:
    def __init__(self, pbar):
        self._p = pbar
    def __call__(self, study, trial):
        self._p.update(1)
        try:
            self._p.set_postfix_str(f'best={study.best_value:.4f}')
        except Exception:
            pass

# ── Boucle principale Optuna ─────────────────────────────────
best_params  = {}   # [model_name][pos] → dict de params
final_models = {}   # [model_name][pos] → modèle entraîné

for mname, mabbr, obj_fn, builder in MODEL_CFG:
    best_params[mname]  = {}
    final_models[mname] = {}
    print(f"▶ {mname}")

    for pos in POSITIONS:
        feats = [f for f in FEATURES_BY_POS[pos] if f in df.columns]
        d_tr  = df_otr[df_otr['position'] == pos]
        d_val = df_oval[df_oval['position'] == pos]

        if len(d_tr) < 30:
            print(f"  [{pos}] skip — données insuffisantes")
            continue

        Xtr = d_tr[feats].fillna(0).values;  ytr = d_tr['target'].values
        wtr = np.where(ytr >= 8, 1.5, 1.0)
        Xv  = d_val[feats].fillna(0).values; yv  = d_val['target'].values

        cache_study = f'optuna_{mabbr}_{pos}.pkl'

        if os.path.exists(cache_study):
            bp = pickle.load(open(cache_study, 'rb'))
            best_params[mname][pos] = bp
            val_mae = mean_absolute_error(yv, build_xgb(bp).fit(Xtr, ytr).predict(Xv)) \
                      if mname == 'XGBoost' else '?'
            print(f"  [{pos}] ✅ chargé depuis cache")
        else:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            objective = lambda t, _Xtr=Xtr, _ytr=ytr, _wtr=wtr, _Xv=Xv, _yv=yv: \
                obj_fn(t, _Xtr, _ytr, _wtr, _Xv, _yv)

            with tqdm(total=N_TRIALS,
                      desc=f"  [{pos}] {mname:10s}",
                      ncols=72, leave=True) as pbar:
                study.optimize(objective, n_trials=N_TRIALS,
                               callbacks=[_TqdmCB(pbar)])

            bp = study.best_params
            best_params[mname][pos] = bp
            pickle.dump(bp, open(cache_study, 'wb'))
            print(f"  [{pos}] best_val_MAE={study.best_value:.4f}  {bp}")

    print()

# ============================================================
# SECTION 4 — Entraînement final + Ensemble (médiane)
# ============================================================
print("=" * 58)
print("  SECTION 4 — ENTRAÎNEMENT FINAL + ENSEMBLE")
print("=" * 58)

model_preds = {mn: [] for mn, *_ in MODEL_CFG}
model_preds['Ensemble'] = []

for mname, mabbr, _, builder in MODEL_CFG:
    for pos in POSITIONS:
        if pos not in best_params.get(mname, {}):
            continue

        feats   = [f for f in FEATURES_BY_POS[pos] if f in df.columns]
        bp      = best_params[mname][pos]
        d_ftr   = df_ftr[df_ftr['position'] == pos]
        d_te    = df_te[df_te['position']   == pos]

        if len(d_ftr) < 30 or len(d_te) == 0:
            continue

        Xftr = d_ftr[feats].fillna(0).values;  yftr = d_ftr['target'].values
        wftr = np.where(yftr >= 8, 1.5, 1.0)
        Xte  = d_te[feats].fillna(0).values

        cache_model = f'best_{mabbr}_{pos}.pkl'

        if os.path.exists(cache_model):
            model = pickle.load(open(cache_model, 'rb'))
            print(f"  [{mname}][{pos}] ✅ modèle chargé depuis cache")
        else:
            print(f"  [{mname}][{pos}] fit sur {len(d_ftr):,} lignes ...",
                  end=' ', flush=True)
            model = builder(bp)
            try:
                model.fit(Xftr, yftr, sample_weight=wftr)
            except Exception:
                model.fit(Xftr, yftr)
            pickle.dump(model, open(cache_model, 'wb'))
            print("OK")

        final_models[mname][pos] = model

        preds = model.predict(Xte)
        chunk = d_te[['GW', 'name', 'position', 'target']].copy()
        chunk['y_pred'] = preds
        model_preds[mname].append(chunk)

print()

# ── Ensemble : médiane des 3 prédictions ─────────────────────
print("  Calcul ensemble (médiane) ...", end=' ', flush=True)
for pos in POSITIONS:
    d_te_pos = df_te[df_te['position'] == pos].reset_index(drop=True)
    if len(d_te_pos) == 0:
        continue

    feats    = [f for f in FEATURES_BY_POS[pos] if f in df.columns]
    Xte      = d_te_pos[feats].fillna(0).values
    pos_preds = []

    for mname in ['XGBoost', 'LightGBM', 'CatBoost']:
        if pos in final_models.get(mname, {}):
            pos_preds.append(final_models[mname][pos].predict(Xte))

    if not pos_preds:
        continue

    ens_preds = np.median(np.vstack(pos_preds), axis=0) \
                if len(pos_preds) > 1 else pos_preds[0]

    chunk = d_te_pos[['GW', 'name', 'position', 'target']].copy()
    chunk['y_pred'] = ens_preds
    model_preds['Ensemble'].append(chunk)

print("OK\n")

# ============================================================
# SECTION 5 — Métriques + Rapport
# ============================================================
print("=" * 58)
print("  SECTION 5 — ÉVALUATION FINALE (test GW 33+)")
print("=" * 58)


def compute_metrics(preds_list: list) -> dict:
    if not preds_list:
        return {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Top10': np.nan}
    df_all = pd.concat(preds_list, ignore_index=True)
    y_true = df_all['target'].values
    y_pred = df_all['y_pred'].values
    return {
        'MAE'  : mean_absolute_error(y_true, y_pred),
        'RMSE' : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2'   : r2_score(y_true, y_pred),
        'Top10': top10_accuracy(df_all),
    }


results = {}
for label in ['XGBoost', 'LightGBM', 'CatBoost', 'Ensemble']:
    results[label] = compute_metrics(model_preds[label])

for label, m in results.items():
    print(f"  {label:12s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"R²={m['R2']:.4f}  Top-10={m['Top10']:.1f}%")

# ── Meilleur modèle ───────────────────────────────────────────
best_label = min(results, key=lambda k: results[k]['MAE'])
best_m     = results[best_label]

t_total = time.time() - t0_global
print(f"\nTemps total : {t_total/60:.1f} min\n")

# ── Tableau comparatif ────────────────────────────────────────
W = 61
sep = "=" * W
dash = "-" * W
hdr  = f"{'':22s}  {'MAE':>6}  {'RMSE':>6}  {'R²':>6}  {'Top-10':>7}"

print(sep)
print("         RECHERCHE DU MEILLEUR MODELE FPL")
print(sep)
print(hdr)
print(dash)
print(f"{'Baseline RF':22s}  {BASELINE['MAE']:>6.2f}  {BASELINE['RMSE']:>6.2f}  "
      f"{BASELINE['R2']:>6.2f}  {BASELINE['Top10']:>6.0f}%")
print(f"{'XGBoost V3':22s}  {PREV_V3['MAE']:>6.2f}  {PREV_V3['RMSE']:>6.2f}  "
      f"{PREV_V3['R2']:>6.2f}  {PREV_V3['Top10']:>6.0f}%")
print(dash)
for label in ['XGBoost', 'LightGBM', 'CatBoost']:
    m = results[label]
    tag = f"{label}+Optuna"
    print(f"{tag:22s}  {m['MAE']:>6.2f}  {m['RMSE']:>6.2f}  "
          f"{m['R2']:>6.2f}  {m['Top10']:>6.0f}%")
print(dash)
ens = results['Ensemble']
print(f"{'ENSEMBLE (mediane)':22s}  {ens['MAE']:>6.2f}  {ens['RMSE']:>6.2f}  "
      f"{ens['R2']:>6.2f}  {ens['Top10']:>6.0f}%")
print(sep)
print(f"Meilleur modele : {best_label}")

mae_gain   = (BASELINE['MAE']   - best_m['MAE'])  / BASELINE['MAE']  * 100
r2_gain    = (best_m['R2']      - BASELINE['R2']) / max(abs(BASELINE['R2']), 1e-6) * 100
top10_gain = best_m['Top10'] - BASELINE['Top10']
print(f"Gain vs Baseline  : MAE -{mae_gain:.1f}%  /  "
      f"R2 {r2_gain:+.1f}%  /  Top-10 {top10_gain:+.0f}%")
print(sep)

# ============================================================
# SECTION 6 — Rapport visuel best_model_report.png
# ============================================================
print("\nGeneration best_model_report.png ...")

labels_plot  = ['Baseline', 'V3', 'XGB+Opt', 'LGB+Opt', 'CAT+Opt', 'Ensemble']
mae_vals  = [BASELINE['MAE'],  PREV_V3['MAE'],
             results['XGBoost']['MAE'],  results['LightGBM']['MAE'],
             results['CatBoost']['MAE'], results['Ensemble']['MAE']]
r2_vals   = [BASELINE['R2'],   PREV_V3['R2'],
             results['XGBoost']['R2'],   results['LightGBM']['R2'],
             results['CatBoost']['R2'],  results['Ensemble']['R2']]
top10_vals= [BASELINE['Top10'],PREV_V3['Top10'],
             results['XGBoost']['Top10'],results['LightGBM']['Top10'],
             results['CatBoost']['Top10'],results['Ensemble']['Top10']]

colors = ['#95a5a6', '#7f8c8d', '#3498db', '#2ecc71', '#e74c3c', '#f39c12']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Recherche du Meilleur Modele FPL — V4\n'
             f'Meilleur : {best_label}  |  MAE={best_m["MAE"]:.2f}  '
             f'R²={best_m["R2"]:.2f}  Top-10={best_m["Top10"]:.0f}%',
             fontsize=13, fontweight='bold')

def bar_plot(ax, vals, title, ylabel, is_higher_better=False):
    bars = ax.bar(labels_plot, vals, color=colors, edgecolor='white', linewidth=0.8)
    best_idx = (np.argmax(vals) if is_higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels_plot, rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

bar_plot(axes[0, 0], mae_vals,   'MAE par modele (↓ meilleur)', 'MAE (points)', False)
bar_plot(axes[0, 1], r2_vals,    'R² par modele (↑ meilleur)',  'R²',           True)
bar_plot(axes[1, 0], top10_vals, 'Top-10 Accuracy (↑ meilleur)','Accuracy (%)', True)

# Feature importance du meilleur modele (position MID)
ax4 = axes[1, 1]
try:
    mid_feats = [f for f in FEATURES_BY_POS['MID'] if f in df.columns]
    best_mid_model = final_models.get(best_label, {}).get('MID')

    if best_mid_model is None:
        best_mid_model = final_models.get('Ensemble', {}).get('MID') or \
                         next((final_models[mn].get('MID')
                               for mn in ['XGBoost', 'LightGBM', 'CatBoost']
                               if 'MID' in final_models.get(mn, {})), None)

    if best_mid_model is not None and mid_feats:
        if hasattr(best_mid_model, 'feature_importances_'):
            imp = pd.Series(best_mid_model.feature_importances_, index=mid_feats)
        elif hasattr(best_mid_model, 'get_feature_importance'):
            imp = pd.Series(best_mid_model.get_feature_importance(), index=mid_feats)
        else:
            imp = pd.Series(np.zeros(len(mid_feats)), index=mid_feats)

        imp = imp.sort_values().tail(20)
        cols_bar = ['#2ecc71' if v >= imp.median() else '#95a5a6' for v in imp.values]
        imp.plot(kind='barh', ax=ax4, color=cols_bar)
        ax4.set_title(f'Feature Importance — MID ({best_label})', fontweight='bold')
        ax4.set_xlabel('Importance')
        ax4.grid(axis='x', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnon disponible',
                 ha='center', va='center', transform=ax4.transAxes)
except Exception as e:
    ax4.text(0.5, 0.5, f'Erreur: {e}', ha='center', va='center',
             transform=ax4.transAxes, fontsize=8)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'best_model_report.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Rapport : {out_path}")
print(f"\nFichiers caches crees :")
for f in sorted(os.listdir('.')):
    if f.endswith('.pkl') and ('best_' in f or 'optuna_' in f or 'cache_merged_v3' in f):
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f:<35s}  {size_kb:>8.1f} KB")
