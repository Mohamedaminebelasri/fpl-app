"""
best_model_v5.py  --  XGBoost V5
9 saisons FPL (2016-17 -> 2024-25)
SHAP feature selection (top 20)
Walk-Forward Validation (5 folds)
XGBoost avec params Optuna existants (optuna_XGB_*.pkl)
"""

# ---- IMPORTS & AUTO-INSTALL ------------------------------------------------
import os, sys, warnings, subprocess, pickle, time, io
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def pip_install(*pkgs):
    for p in pkgs:
        try:
            __import__(p.split('[')[0].replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', p, '-q'])

pip_install('xgboost', 'requests', 'shap', 'matplotlib', 'scikit-learn')

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

print("=" * 65)
print("   BEST_MODEL_V5.PY  --  9 Saisons + SHAP + Walk-Forward")
print("=" * 65)

t_global = time.time()

# ---- CONSTANTS -------------------------------------------------------------
SEASONS = [
    '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'
]
SEASON_ORDER = {s: i + 1 for i, s in enumerate(SEASONS)}

BASE_URL     = ("https://raw.githubusercontent.com/vaastav/"
                "Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv")
PLAYERS_URL  = ("https://raw.githubusercontent.com/vaastav/"
                "Fantasy-Premier-League/master/data/{season}/players_raw.csv")

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
POS_MAP = {
    1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD',
    '1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD',
    'GK': 'GK', 'GKP': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD',
    'AM': 'MID', 'Goalkeeper': 'GK', 'GKP': 'GK',
    'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
    'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD', 'ATT': 'FWD',
}

FIXED_PARAMS = dict(n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')

# Walk-forward folds: (test_season, gw_start, gw_end)
WALK_FOLDS = [
    ('2023-24',  1, 10),
    ('2023-24', 11, 20),
    ('2023-24', 21, 38),
    ('2024-25',  1, 19),
    ('2024-25', 20, 38),
]

QUALITY_COLS = [
    'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    'q_xgc_per90', 'q_cs_per90', 'q_saves_per90',
]


# ---- HELPERS ---------------------------------------------------------------
def top10_accuracy(df_eval: pd.DataFrame) -> float:
    hits, n_gw = 0, 0
    for gw in sorted(df_eval['GW'].unique()):
        gw_d = df_eval[df_eval['GW'] == gw]
        if len(gw_d) < 10:
            continue
        hits += len(
            set(gw_d.nlargest(10, 'y_pred').index) &
            set(gw_d.nlargest(10, 'target').index)
        )
        n_gw += 1
    return (hits / (n_gw * 10)) * 100 if n_gw > 0 else 0.0


def evaluate(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


def load_optuna_params(pos: str) -> dict:
    path = f'optuna_XGB_{pos}.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2)


def train_pos_model(df_tr: pd.DataFrame, pos: str,
                    features: list, optuna_params: dict,
                    sample_weight: float = 1.5):
    d = df_tr[df_tr['position'] == pos]
    if len(d) < 50:
        return None
    X = d[features].fillna(0).values
    y = d['target'].values
    w = np.where(y >= 8, sample_weight, 1.0)
    params = {**optuna_params, **FIXED_PARAMS}
    mdl = XGBRegressor(**params)
    mdl.fit(X, y, sample_weight=w, verbose=False)
    return mdl


def predict_pos(models: dict, df_te: pd.DataFrame, features: list) -> pd.Series:
    preds = pd.Series(np.nan, index=df_te.index)
    for pos, mdl in models.items():
        if mdl is None:
            continue
        mask = df_te['position'] == pos
        if mask.sum() == 0:
            continue
        X = df_te.loc[mask, features].fillna(0).values
        preds.loc[mask] = mdl.predict(X)
    return preds


# =============================================================================
# SECTION 1  --  Download 9 seasons
# =============================================================================
print("\n[1/6] Telechargement des 9 saisons ...")
CACHE_9S = 'cache_9seasons.pkl'


def fetch_player_positions(season: str) -> pd.DataFrame:
    """Download players_raw.csv and return df with element, position, team."""
    url = PLAYERS_URL.format(season=season)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df_p = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        # element id
        id_col = 'id' if 'id' in df_p.columns else 'code'
        df_p['_elem_id'] = pd.to_numeric(df_p[id_col], errors='coerce')
        # position
        if 'element_type' in df_p.columns:
            df_p['_pos'] = df_p['element_type'].map(
                lambda x: POS_MAP.get(str(x).strip(), None))
        elif 'position' in df_p.columns:
            df_p['_pos'] = df_p['position'].map(
                lambda x: POS_MAP.get(str(x).strip(), None))
        else:
            df_p['_pos'] = None
        # team
        team_col = None
        for tc in ['team', 'team_name', 'team_code']:
            if tc in df_p.columns:
                team_col = tc
                break
        df_p['_team'] = df_p[team_col].astype(str) if team_col else f'unk_{season}'
        return df_p[['_elem_id', '_pos', '_team']].dropna(subset=['_elem_id'])
    except Exception as e:
        print(f"      players_raw error ({e})")
        return pd.DataFrame(columns=['_elem_id', '_pos', '_team'])


def standardise(df_s: pd.DataFrame, season: str,
                player_info: pd.DataFrame) -> pd.DataFrame:
    # ---- position -----------------------------------------------------------
    if 'position' not in df_s.columns or df_s['position'].isna().all():
        if len(player_info) > 0 and 'element' in df_s.columns:
            elem_to_pos  = dict(zip(player_info['_elem_id'], player_info['_pos']))
            elem_to_team = dict(zip(player_info['_elem_id'], player_info['_team']))
            df_s['element'] = pd.to_numeric(df_s['element'], errors='coerce')
            df_s['position'] = df_s['element'].map(elem_to_pos)
            if 'team' not in df_s.columns:
                df_s['team'] = df_s['element'].map(elem_to_team).fillna(f'unk_{season}')
    else:
        df_s['position'] = df_s['position'].map(
            lambda x: POS_MAP.get(str(x).strip(), None))

    # ---- value / now_cost ---------------------------------------------------
    if 'value' not in df_s.columns:
        if 'now_cost' in df_s.columns:
            df_s['value'] = df_s['now_cost']
        else:
            df_s['value'] = 0.0

    # ---- was_home -----------------------------------------------------------
    if 'was_home' in df_s.columns:
        df_s['was_home'] = df_s['was_home'].map(
            {True: 1, False: 0, 'True': 1, 'False': 0,
             1: 1, 0: 0, '1': 1, '0': 0}
        ).fillna(0).astype(int)
    else:
        df_s['was_home'] = 0

    # ---- numeric cols -------------------------------------------------------
    num_cols = [
        'total_points', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'goals_conceded', 'bonus', 'bps', 'influence', 'creativity', 'threat',
        'ict_index', 'value', 'selected', 'transfers_in', 'transfers_out',
        'opponent_team', 'saves',
        'expected_goals', 'expected_assists', 'expected_goals_conceded',
    ]
    for c in num_cols:
        if c in df_s.columns:
            df_s[c] = pd.to_numeric(df_s[c], errors='coerce').fillna(0)
        else:
            df_s[c] = 0.0

    # ---- team ---------------------------------------------------------------
    if 'team' not in df_s.columns:
        df_s['team'] = f'unk_{season}'
    else:
        df_s['team'] = df_s['team'].astype(str).str.strip()

    # ---- opponent_team as string --------------------------------------------
    df_s['opponent_team'] = (
        pd.to_numeric(df_s['opponent_team'], errors='coerce')
        .fillna(0).astype(int).astype(str)
    )

    df_s['season'] = season
    return df_s


if os.path.exists(CACHE_9S):
    print(f"   Cache '{CACHE_9S}' trouve, chargement ...")
    df_all = pickle.load(open(CACHE_9S, 'rb'))
else:
    frames = []
    for i, season in enumerate(SEASONS, 1):
        try:
            print(f"   [{i}/{len(SEASONS)}] {season} ...", end=' ', flush=True)
            resp = requests.get(BASE_URL.format(season=season), timeout=30)
            resp.raise_for_status()
            df_s = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            needs_pos = 'position' not in df_s.columns or df_s['position'].isna().all()
            player_info = fetch_player_positions(season) if needs_pos else pd.DataFrame()
            df_s = standardise(df_s, season, player_info)
            n_before = len(df_s)
            df_s = df_s[df_s['position'].isin(POSITIONS)].copy()
            print(f"{len(df_s):,} lignes  (pos OK: {len(df_s):,}/{n_before:,})")
            frames.append(df_s)
        except Exception as e:
            print(f"ERREUR: {e}")

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    df_all['season_order'] = df_all['season'].map(SEASON_ORDER).fillna(0).astype(int)
    pickle.dump(df_all, open(CACHE_9S, 'wb'))
    print(f"   Sauvegarde '{CACHE_9S}': {len(df_all):,} lignes")

print(f"   Dataset: {len(df_all):,} lignes | saisons: {sorted(df_all['season'].unique())}")


# =============================================================================
# SECTION 2  --  Feature Engineering
# =============================================================================
print("\n[2/6] Feature Engineering ...")
CACHE_FEAT = 'cache_9seasons_features.pkl'

if os.path.exists(CACHE_FEAT):
    print(f"   Cache '{CACHE_FEAT}' trouve, chargement ...")
    df_feat = pickle.load(open(CACHE_FEAT, 'rb'))
else:
    df = df_all.copy()
    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

    def roll(col, window, func='mean'):
        grp = df.groupby(['name', 'season'])[col]
        if func == 'mean':
            return grp.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            ).fillna(0)
        return grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        ).fillna(0)

    print("   Rolling features ...")
    df['pts_rolling_1']     = roll('total_points', 1)
    df['pts_rolling_3']     = roll('total_points', 3)
    df['pts_rolling_5']     = roll('total_points', 5)
    df['pts_rolling_10']    = roll('total_points', 10)
    df['mins_rolling_3']    = roll('minutes', 3)
    df['bps_rolling_3']     = roll('bps', 3)
    df['goals_rolling_5']   = roll('goals_scored', 5)
    df['assists_rolling_5'] = roll('assists', 5)
    df['xg_rolling_3']      = roll('expected_goals', 3)
    df['xg_rolling_5']      = roll('expected_goals', 5)
    df['xa_rolling_3']      = roll('expected_assists', 3)
    df['xgc_rolling_5']     = roll('expected_goals_conceded', 5)

    print("   Interaction features ...")
    df['form_momentum'] = df['pts_rolling_3'] - df['pts_rolling_10']
    df['value_score']   = (
        df['pts_rolling_5'] / (df['value'].clip(lower=0.1) + 0.1)
    ).clip(0, 10)
    pts_std5             = roll('total_points', 5, func='std')
    df['consistency']    = (1 - pts_std5 / (df['pts_rolling_5'] + 0.1)).clip(-2, 1)
    df['big_game_flag']  = np.where(df['was_home'] == 1, 1.15, 0.85)
    df['net_transfers']  = df['transfers_in'] - df['transfers_out']
    df['mins_pct']       = (df['mins_rolling_3'] / 90.0).clip(0, 1)

    print("   Contextual features ...")
    df['is_endseason']   = (df['GW'] >= 33).astype(int)
    df['is_earlyseason'] = (df['GW'] <= 5).astype(int)
    df['gw_normalized']  = df['GW'] / 38.0
    if 'season_order' not in df.columns:
        df['season_order'] = df['season'].map(SEASON_ORDER).fillna(0).astype(int)

    print("   Team encoding features ...")
    df_s2 = df.sort_values(['team', 'season', 'GW']).copy()

    tgw = df_s2.groupby(['season', 'team', 'GW']).agg(
        _team_pts   =('total_points', 'mean'),
        _team_goals =('goals_scored',  'sum'),
        _team_cs    =('clean_sheets',  'max'),
    ).reset_index()
    for src, dst in [('_team_pts',   'team_avg_pts_scored'),
                     ('_team_goals', 'team_goals_per_game'),
                     ('_team_cs',    'team_clean_sheet_rate')]:
        tgw[dst] = (
            tgw.groupby(['team', 'season'])[src]
               .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
    team_enc = tgw[['season', 'team', 'GW',
                    'team_avg_pts_scored', 'team_goals_per_game',
                    'team_clean_sheet_rate']]
    df = df.merge(team_enc, on=['season', 'team', 'GW'], how='left')
    for c in ['team_avg_pts_scored', 'team_goals_per_game', 'team_clean_sheet_rate']:
        df[c] = df[c].fillna(0)

    ogc = (df.groupby(['season', 'opponent_team', 'GW'])['goals_scored']
             .sum().reset_index().rename(columns={'goals_scored': '_gc'}))
    ogc = ogc.sort_values(['opponent_team', 'season', 'GW'])
    ogc['opponent_goals_conceded_avg'] = (
        ogc.groupby(['opponent_team', 'season'])['_gc']
           .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    ogc_merge = (ogc[['season', 'GW', 'opponent_team', 'opponent_goals_conceded_avg']]
                 .rename(columns={'opponent_team': '_opp_key'}))
    df = df.merge(
        ogc_merge,
        left_on=['season', 'GW', 'opponent_team'],
        right_on=['season', 'GW', '_opp_key'],
        how='left'
    ).drop(columns=['_opp_key'])
    df['opponent_goals_conceded_avg'] = df['opponent_goals_conceded_avg'].fillna(0)

    print("   Quality features (cache_merged_v2.pkl) ...")
    if os.path.exists('cache_merged_v2.pkl'):
        df_v2 = pickle.load(open('cache_merged_v2.pkl', 'rb'))
        q_src = df_v2[['name', 'season', 'GW'] + QUALITY_COLS].copy()
        for c in QUALITY_COLS:
            q_src[c] = pd.to_numeric(q_src[c], errors='coerce').fillna(0)
        df = df.merge(q_src, on=['name', 'season', 'GW'], how='left')
    for c in QUALITY_COLS:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0)

    # Target: next GW total_points
    df = df.sort_values(['name', 'season', 'GW'])
    df['target'] = df.groupby(['name', 'season'])['total_points'].transform(
        lambda x: x.shift(-1)
    )
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(float)

    df_feat = df.reset_index(drop=True)
    pickle.dump(df_feat, open(CACHE_FEAT, 'wb'))
    print(f"   Sauvegarde '{CACHE_FEAT}': {len(df_feat):,} lignes, {df_feat.shape[1]} colonnes")

print(f"   Shape: {df_feat.shape} | saisons: {sorted(df_feat['season'].unique())}")

ALL_FEATURES = [f for f in [
    # Base stats
    'minutes', 'bps', 'ict_index', 'value', 'was_home', 'season_order',
    'goals_scored', 'assists', 'clean_sheets', 'bonus',
    'influence', 'creativity', 'threat',
    'selected', 'transfers_in', 'transfers_out', 'net_transfers',
    'saves', 'mins_pct',
    # Rolling
    'pts_rolling_1', 'pts_rolling_3', 'pts_rolling_5', 'pts_rolling_10',
    'mins_rolling_3', 'bps_rolling_3', 'goals_rolling_5', 'assists_rolling_5',
    'xg_rolling_3', 'xg_rolling_5', 'xa_rolling_3', 'xgc_rolling_5',
    # Interaction
    'form_momentum', 'value_score', 'consistency', 'big_game_flag',
    # Team encoding
    'team_avg_pts_scored', 'team_clean_sheet_rate',
    'team_goals_per_game', 'opponent_goals_conceded_avg',
    # Contextual
    'is_endseason', 'is_earlyseason', 'gw_normalized',
    # Quality (Understat -- available 2023-24 + 2024-25)
    'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    'q_xgc_per90', 'q_cs_per90', 'q_saves_per90',
] if f in df_feat.columns]
print(f"   Candidats SHAP: {len(ALL_FEATURES)} features")


# =============================================================================
# SECTION 3  --  SHAP Feature Selection (top 20)
# =============================================================================
print("\n[3/6] Selection SHAP des top 20 features ...")
CACHE_SHAP = 'shap_top_features.pkl'

if os.path.exists(CACHE_SHAP):
    print(f"   Cache '{CACHE_SHAP}' trouve, chargement ...")
    top20_features = pickle.load(open(CACHE_SHAP, 'rb'))
else:
    mask_shap = (df_feat['season'] != '2024-25') | (df_feat['GW'] <= 28)
    df_shap_pool = df_feat[mask_shap].copy()

    np.random.seed(42)
    n_sample = max(5000, int(0.20 * len(df_shap_pool)))
    idx = np.random.choice(len(df_shap_pool), size=n_sample, replace=False)
    X_s = df_shap_pool[ALL_FEATURES].fillna(0).values[idx]
    y_s = df_shap_pool['target'].values[idx]

    print(f"   SHAP: {n_sample:,} samples x {len(ALL_FEATURES)} features ...")
    t0 = time.time()
    quick_model = XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, **FIXED_PARAMS
    )
    quick_model.fit(X_s, y_s, verbose=False)

    try:
        explainer      = shap.TreeExplainer(quick_model)
        shap_vals      = explainer.shap_values(X_s, check_additivity=False)
        shap_imp       = np.abs(shap_vals).mean(axis=0)
        top20_idx      = np.argsort(shap_imp)[::-1][:20]
        top20_features = [ALL_FEATURES[i] for i in top20_idx]
        method = 'SHAP'
    except Exception as e:
        print(f"   SHAP erreur ({e}), fallback XGB importance")
        imp = quick_model.feature_importances_
        top20_idx      = np.argsort(imp)[::-1][:20]
        top20_features = [ALL_FEATURES[i] for i in top20_idx]
        method = 'XGB_importance'

    print(f"   Methode: {method} | Duree: {time.time() - t0:.1f}s")
    pickle.dump(top20_features, open(CACHE_SHAP, 'wb'))
    print(f"   Sauvegarde '{CACHE_SHAP}'")

print(f"   Top 20 features:")
for i, f in enumerate(top20_features, 1):
    print(f"     {i:2d}. {f}")


# =============================================================================
# SECTION 4  --  Walk-Forward Validation
# =============================================================================
print("\n[4/6] Walk-Forward Validation (5 folds) ...")

wf_results   = []
all_preds_wf = []

for fold_i, (test_season, gw_start, gw_end) in enumerate(WALK_FOLDS, 1):
    mask_train = (
        (df_feat['season'] != test_season) |
        (df_feat['GW'] < gw_start)
    )
    mask_test = (
        (df_feat['season'] == test_season) &
        (df_feat['GW'] >= gw_start) &
        (df_feat['GW'] <= gw_end)
    )
    df_tr = df_feat[mask_train].copy()
    df_te = df_feat[mask_test].copy()

    if len(df_te) < 50:
        print(f"   Fold {fold_i}: pas assez de donnees test, skip")
        continue

    fold_models = {}
    for pos in POSITIONS:
        opt_params = load_optuna_params(pos)
        fold_models[pos] = train_pos_model(df_tr, pos, top20_features, opt_params)

    df_te = df_te.copy()
    df_te['y_pred'] = predict_pos(fold_models, df_te, top20_features)
    df_te_clean = df_te.dropna(subset=['y_pred'])

    if len(df_te_clean) == 0:
        continue

    mae, rmse, r2 = evaluate(df_te_clean['target'], df_te_clean['y_pred'])
    top10 = top10_accuracy(df_te_clean.reset_index(drop=True))

    wf_results.append({'fold': fold_i, 'season': test_season,
                       'gw_range': f'{gw_start}-{gw_end}',
                       'n_test': len(df_te_clean),
                       'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Top10': top10})
    all_preds_wf.append(df_te_clean)

    print(f"   Fold {fold_i} ({test_season} GW{gw_start}-{gw_end}): "
          f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  "
          f"Top-10={top10:.1f}%  (n={len(df_te_clean):,})")

if wf_results:
    wf_df   = pd.DataFrame(wf_results)
    wf_mae  = wf_df['MAE'].mean()
    wf_rmse = wf_df['RMSE'].mean()
    wf_r2   = wf_df['R2'].mean()
    wf_top10= wf_df['Top10'].mean()
    print(f"\n   -- Walk-Forward Moyen --")
    print(f"   MAE={wf_mae:.4f}  RMSE={wf_rmse:.4f}  R2={wf_r2:.4f}  Top-10={wf_top10:.1f}%")
else:
    wf_mae = wf_rmse = wf_r2 = wf_top10 = float('nan')
    print("   Aucun fold valide.")


# =============================================================================
# SECTION 5  --  Final Training
# =============================================================================
print("\n[5/6] Entrainement final (train GW1-32, test GW33+ de 2024-25) ...")

mask_train_final = (
    (df_feat['season'] != '2024-25') |
    (df_feat['GW'] <= 32)
)
mask_test_final = (
    (df_feat['season'] == '2024-25') &
    (df_feat['GW'] >= 33)
)

df_train_final = df_feat[mask_train_final].copy()
df_test_final  = df_feat[mask_test_final].copy()
print(f"   Train: {len(df_train_final):,} | Test GW33+: {len(df_test_final):,}")

final_models = {}
for pos in POSITIONS:
    opt_params = load_optuna_params(pos)
    mdl = train_pos_model(df_train_final, pos, top20_features, opt_params)
    if mdl is not None:
        final_models[pos] = mdl
        pickle.dump(mdl, open(f'model_v5_{pos}.pkl', 'wb'))
        n_pos = (df_train_final['position'] == pos).sum()
        print(f"   {pos}: {n_pos:,} samples -> model_v5_{pos}.pkl")
    else:
        print(f"   {pos}: pas assez de donnees")

if len(df_test_final) >= 50:
    df_test_final = df_test_final.copy()
    df_test_final['y_pred'] = predict_pos(final_models, df_test_final, top20_features)
    df_eval_final = df_test_final.dropna(subset=['y_pred'])
    final_mae, final_rmse, final_r2 = evaluate(
        df_eval_final['target'], df_eval_final['y_pred'])
    final_top10 = top10_accuracy(df_eval_final.reset_index(drop=True))
    print(f"\n   -- Score Final (2024-25 GW33+) --")
    print(f"   MAE={final_mae:.4f}  RMSE={final_rmse:.4f}  "
          f"R2={final_r2:.4f}  Top-10={final_top10:.1f}%")
elif all_preds_wf:
    df_all_wf = pd.concat(all_preds_wf, ignore_index=True)
    final_mae, final_rmse, final_r2 = evaluate(
        df_all_wf['target'], df_all_wf['y_pred'])
    final_top10 = top10_accuracy(df_all_wf)
    print(f"   (GW33+ indisponible -> tous les folds WF)")
    print(f"   MAE={final_mae:.4f}  RMSE={final_rmse:.4f}  "
          f"R2={final_r2:.4f}  Top-10={final_top10:.1f}%")
else:
    final_mae = final_rmse = final_r2 = final_top10 = float('nan')
    print("   Aucune donnee de test finale.")


# =============================================================================
# SECTION 6  --  Report
# =============================================================================
print("\n[6/6] Generation du rapport ...")

comparison = pd.DataFrame([
    {'Version': 'XGBoost V3',    'Saisons': 2, 'Features': 38,
     'MAE': 0.9500, 'RMSE': 1.8600, 'R2': 0.28, 'Top10': 6.0},
    {'Version': 'Ensemble V4',   'Saisons': 2, 'Features': 42,
     'MAE': 1.0876, 'RMSE': 2.0470, 'R2': 0.29, 'Top10': 8.0},
    {'Version': 'XGBoost V5 WF', 'Saisons': 9, 'Features': 'TOP20',
     'MAE': round(wf_mae,   4), 'RMSE': round(wf_rmse,   4),
     'R2':  round(wf_r2,    4), 'Top10': round(wf_top10, 1)},
    {'Version': 'XGBoost V5',    'Saisons': 9, 'Features': 'TOP20',
     'MAE': round(final_mae,  4), 'RMSE': round(final_rmse,  4),
     'R2':  round(final_r2,   4), 'Top10': round(final_top10, 1)},
])

print("\n" + "=" * 72)
print(comparison.to_string(index=False))
print("=" * 72)

# ---- PNG --------------------------------------------------------------------
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("XGBoost V5 -- 9 Saisons + SHAP + Walk-Forward",
                 fontsize=14, fontweight='bold')

    # 1) Comparison table
    ax = axes[0, 0]
    ax.axis('off')
    tbl_data = comparison[['Version', 'MAE', 'RMSE', 'R2', 'Top10']].copy()
    tbl_data.columns = ['Version', 'MAE', 'RMSE', 'R2', 'Top-10%']
    tbl = ax.table(cellText=tbl_data.values, colLabels=tbl_data.columns,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    # highlight best MAE row (skip header row 0)
    try:
        best_mae_row = int(tbl_data['MAE'].astype(float).idxmin()) + 1
        for j in range(len(tbl_data.columns)):
            tbl[best_mae_row, j].set_facecolor('#d4edda')
    except Exception:
        pass
    ax.set_title('Comparaison des modeles', fontweight='bold', pad=12)

    # 2) Walk-Forward MAE per fold
    ax = axes[0, 1]
    if wf_results:
        fold_labels = [f"F{r['fold']}\n{r['season']}\nGW{r['gw_range']}"
                       for r in wf_results]
        fold_maes   = [r['MAE']   for r in wf_results]
        bars = ax.bar(fold_labels, fold_maes, color='steelblue', alpha=0.75)
        ax.axhline(wf_mae,  color='red',    linestyle='--', lw=1.5, label=f'Moy={wf_mae:.3f}')
        ax.axhline(0.95,    color='green',  linestyle=':',  lw=1.2, label='V3=0.950')
        ax.axhline(1.0876,  color='orange', linestyle=':',  lw=1.2, label='V4=1.088')
        for bar, v in zip(bars, fold_maes):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_title('MAE Walk-Forward par fold', fontweight='bold')
        ax.set_ylabel('MAE')
        ax.legend(fontsize=7)
        ax.set_ylim(0, max(fold_maes) * 1.35)
    else:
        ax.text(0.5, 0.5, 'No WF results', ha='center', va='center')
        ax.set_title('Walk-Forward MAE')

    # 3) Top-20 feature importance (averaged over final models)
    ax = axes[1, 0]
    if final_models and top20_features:
        imp_sum = np.zeros(len(top20_features))
        n_m = 0
        for pos, mdl in final_models.items():
            if mdl is None:
                continue
            feat_names_mdl = (
                list(mdl.feature_names_in_)
                if hasattr(mdl, 'feature_names_in_') else top20_features
            )
            imp_map = dict(zip(feat_names_mdl, mdl.feature_importances_))
            for i, f in enumerate(top20_features):
                imp_sum[i] += imp_map.get(f, 0)
            n_m += 1
        if n_m:
            imp_sum /= n_m
        sorted_idx = np.argsort(imp_sum)
        ax.barh([top20_features[i] for i in sorted_idx],
                imp_sum[sorted_idx], color='darkorange', alpha=0.75)
        ax.set_title('Feature importance (moy. 4 pos.)', fontweight='bold')
        ax.set_xlabel('Importance')
    else:
        ax.text(0.5, 0.5, 'No models', ha='center', va='center')
        ax.set_title('Feature Importance')

    # 4) MAE & RMSE comparison
    ax = axes[1, 1]
    versions  = ['V3', 'V4 Ens.', 'V5 WF', 'V5']
    mae_vals  = [0.9500, 1.0876, wf_mae,  final_mae]
    rmse_vals = [1.8600, 2.0470, wf_rmse, final_rmse]
    x = np.arange(len(versions))
    w = 0.35
    b1 = ax.bar(x - w / 2, mae_vals,  w, label='MAE',  color='steelblue', alpha=0.8)
    b2 = ax.bar(x + w / 2, rmse_vals, w, label='RMSE', color='tomato',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel('Erreur')
    ax.set_title('MAE & RMSE par version', fontweight='bold')
    ax.legend()
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('best_model_v5_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Rapport sauvegarde: best_model_v5_report.png")
except Exception as e:
    print(f"   Erreur rapport PNG: {e}")

total_time = time.time() - t_global
print(f"\n   Duree totale: {total_time / 60:.1f} min")
print("\n" + "=" * 72)
print("   TERMINE  --  XGBoost V5")
print("=" * 72)
