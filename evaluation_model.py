"""
evaluation_model.py — Évaluation améliorée (v3)
XGBoost × 4 positions + rolling 3/5 + features adversaire + form_trend
Données : vaastav/Fantasy-Premier-League (2023-24 + 2024-25)
"""

import os, sys, re, warnings, subprocess, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from io import StringIO

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
except ImportError:
    print("Installation xgboost ...")
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'xgboost', '--quiet'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASELINE = {'MAE': 1.03, 'RMSE': 2.01, 'R2': 0.31, 'Top10': 10.0}
V1V2     = {'MAE': 0.96, 'RMSE': 1.89, 'R2': 0.25, 'Top10': 6.0}
HEADERS  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

BASE_URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data'
GW_URLS  = {
    '2023-24': f'{BASE_URL}/2023-24/gws/merged_gw.csv',
    '2024-25': f'{BASE_URL}/2024-25/gws/merged_gw.csv',
}
GW_COLS = [
    'name', 'position', 'GW',
    'minutes', 'total_points', 'bps', 'bonus',
    'influence', 'creativity', 'threat', 'ict_index',
    'value', 'was_home', 'opponent_team',
    'saves', 'clean_sheets', 'goals_scored', 'assists',
    'expected_goals', 'expected_assists', 'expected_goals_conceded',
]
RAW_URLS = {
    '2023-24': f'{BASE_URL}/2023-24/players_raw.csv',
    '2024-25': f'{BASE_URL}/2024-25/players_raw.csv',
}
RAW_COLS = [
    'first_name', 'second_name',
    'expected_goals_per_90', 'expected_assists_per_90',
    'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90',
    'clean_sheets_per_90', 'saves_per_90', 'minutes',
]
QUALITY_FEATS = [
    'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    'q_xgc_per90', 'q_cs_per90', 'q_saves_per90',
]

# ============================================================
# SECTION 1 — Chargement des données merged_gw
# ============================================================
print("=" * 52)
print("  SECTION 1 — CHARGEMENT DES DONNÉES")
print("=" * 52)

if os.path.exists('cache_data.pkl'):
    df = pickle.load(open('cache_data.pkl', 'rb'))
    print("✅ Données chargées depuis cache")
else:
    frames = []
    for season, url in GW_URLS.items():
        print(f"merged_gw {season} ...", end=' ', flush=True)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            df_s = pd.read_csv(StringIO(r.text))
            df_s['season'] = season
            cols = [c for c in GW_COLS if c in df_s.columns] + ['season']
            frames.append(df_s[cols].copy())
            print(f"OK  ({len(df_s):,} lignes, {df_s['GW'].nunique()} GWs)")
        except Exception as e:
            print(f"ERREUR : {e}"); sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    NUM_COLS = [
        'minutes', 'total_points', 'bps', 'bonus',
        'influence', 'creativity', 'threat', 'ict_index',
        'value', 'opponent_team', 'saves', 'clean_sheets',
        'goals_scored', 'assists',
        'expected_goals', 'expected_assists', 'expected_goals_conceded',
    ]
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['was_home'] = (
        df['was_home'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        .fillna(0).astype(int)
    )
    pickle.dump(df, open('cache_data.pkl', 'wb'))
    print("✅ Données téléchargées et sauvegardées")

print(f"Total : {len(df):,} lignes\n")

# ============================================================
# SECTION 2 — Qualité joueur (players_raw per90)
# ============================================================
print("=" * 52)
print("  SECTION 2 — QUALITÉ JOUEUR (players_raw per90)")
print("=" * 52)


def normalize_name(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', s.lower().strip())


def best_match(fpl_name: str, candidates: list[str], cutoff: float = 0.55) -> str | None:
    n = normalize_name(fpl_name)
    norm_cands = [normalize_name(c) for c in candidates]
    for raw, norm in zip(candidates, norm_cands):
        if n in norm or norm in n:
            return raw
    import difflib
    close = difflib.get_close_matches(n, norm_cands, n=1, cutoff=cutoff)
    if close:
        return candidates[norm_cands.index(close[0])]
    return None


if os.path.exists('cache_merged.pkl'):
    df = pickle.load(open('cache_merged.pkl', 'rb'))
    print("✅ Fusion chargée depuis cache")
else:
    raw_dfs = {}
    for season, url in RAW_URLS.items():
        print(f"players_raw {season} ...", end=' ', flush=True)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.raise_for_status()
            df_r = pd.read_csv(StringIO(r.text))
            avail = [c for c in RAW_COLS if c in df_r.columns]
            df_r  = df_r[avail].copy()
            if 'first_name' in df_r.columns and 'second_name' in df_r.columns:
                df_r['full_name'] = (
                    df_r['first_name'].fillna('') + ' ' + df_r['second_name'].fillna('')
                ).str.strip()
            else:
                df_r['full_name'] = ''
            for old, new in [
                ('expected_goals_per_90',           'q_xg_per90'),
                ('expected_assists_per_90',          'q_xa_per90'),
                ('expected_goal_involvements_per_90','q_xgi_per90'),
                ('expected_goals_conceded_per_90',   'q_xgc_per90'),
                ('clean_sheets_per_90',              'q_cs_per90'),
                ('saves_per_90',                     'q_saves_per90'),
            ]:
                if old in df_r.columns:
                    df_r[new] = pd.to_numeric(df_r[old], errors='coerce').fillna(0)
                else:
                    df_r[new] = 0.0
            raw_dfs[season] = df_r[['full_name'] + QUALITY_FEATS]
            print(f"OK  ({len(df_r)} joueurs)")
        except Exception as e:
            print(f"IGNORÉ ({e})")

    for q in QUALITY_FEATS:
        df[q] = 0.0
    for target_season, src_season in {'2024-25': '2023-24'}.items():
        if src_season not in raw_dfs:
            continue
        df_q      = raw_dfs[src_season]
        name_pool = df_q['full_name'].tolist()
        fpl_names = df.loc[df['season'] == target_season, 'name'].unique()
        name_map  = {}
        for fpl_n in fpl_names:
            m = best_match(fpl_n, name_pool)
            if m:
                name_map[fpl_n] = m
        for fpl_n, raw_n in name_map.items():
            row_q       = df_q[df_q['full_name'] == raw_n].iloc[0]
            target_mask = (df['season'] == target_season) & (df['name'] == fpl_n)
            for q in QUALITY_FEATS:
                df.loc[target_mask, q] = float(row_q[q])

    pickle.dump(df, open('cache_merged.pkl', 'wb'))
    print("✅ Fusion sauvegardée")

matched = (df['q_xg_per90'] > 0).sum()
print(f"Lignes avec qualité per90 : {matched:,}/{len(df):,}  ({100*matched/len(df):.0f}%)\n")

# ============================================================
# SECTION 2.5 — FIX 1+2+3 : rolling 3/5, adversaire, endseason
# ============================================================
print("=" * 52)
print("  SECTION 2.5 — FEATURES V3")
print("=" * 52)

if os.path.exists('cache_merged_v2.pkl'):
    df = pickle.load(open('cache_merged_v2.pkl', 'rb'))
    print("✅ Features V3 chargées depuis cache")
else:
    print("Calcul features V3 ...", flush=True)
    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

    # FIX 1 — Rolling 3 et 5 GWs (shift(1) = zéro fuite)
    def rolling_n(col: str, n: int) -> pd.Series:
        return (
            df.groupby(['name', 'season'])[col]
            .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
        )

    df['xG_rolling_3']   = rolling_n('expected_goals', 3)
    df['xG_rolling_5']   = rolling_n('expected_goals', 5)
    df['pts_rolling_3']  = rolling_n('total_points', 3)
    df['pts_rolling_5']  = rolling_n('total_points', 5)
    df['form_trend']     = df['pts_rolling_3'] - df['pts_rolling_5']
    df['mins_rolling_3'] = rolling_n('minutes', 3)

    # FIX 3 — Flag fin de saison
    df['is_endseason'] = (df['GW'] >= 33).astype(int)

    # FIX 2 — Features équipe adverse (depuis cache_data.pkl)
    df_raw = pickle.load(open('cache_data.pkl', 'rb'))

    # Goals conceded by team T per GW = goals_scored by all players facing T
    team_gc = (
        df_raw.groupby(['season', 'opponent_team', 'GW'])['goals_scored']
        .sum().reset_index()
        .rename(columns={'opponent_team': 'team', 'goals_scored': 'goals_against'})
        .sort_values(['team', 'season', 'GW'])
        .reset_index(drop=True)
    )

    # Rolling 5 GWs de goals concédés (shift(1) = zéro fuite)
    team_gc['opponent_goals_conceded_avg'] = (
        team_gc.groupby(['team', 'season'])['goals_against']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0)
    )

    # is_top6_opponent : équipes qui ont le moins concédé de buts cumulés (meilleure défense)
    team_gc['cumul_against'] = (
        team_gc.groupby(['team', 'season'])['goals_against']
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    team_gc['is_top6_opponent'] = (
        team_gc.groupby(['season', 'GW'])['cumul_against']
        .rank(ascending=True, method='first') <= 6
    ).astype(int)

    # Jointure sur (season, GW, opponent_team)
    opp_cols = team_gc[['season', 'GW', 'team',
                         'opponent_goals_conceded_avg', 'is_top6_opponent']]
    df = df.merge(
        opp_cols,
        left_on=['season', 'GW', 'opponent_team'],
        right_on=['season', 'GW', 'team'],
        how='left'
    ).drop(columns=['team'])
    df['opponent_goals_conceded_avg'] = df['opponent_goals_conceded_avg'].fillna(0)
    df['is_top6_opponent']            = df['is_top6_opponent'].fillna(0).astype(int)

    # Colonne dupliquée pour double poids DEF/GK
    df['opponent_gc_avg_def'] = df['opponent_goals_conceded_avg']

    df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)
    pickle.dump(df, open('cache_merged_v2.pkl', 'wb'))
    print("✅ Features V3 sauvegardées dans cache_merged_v2.pkl")

print(f"Colonnes : {len(df.columns)}  |  Lignes : {len(df):,}\n")

# ============================================================
# SECTION 3 — Rolling feature engineering (window=4, existant)
# ============================================================
print("=" * 52)
print("  SECTION 3 — FEATURE ENGINEERING")
print("=" * 52)

df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)


def rolling4(col: str) -> pd.Series:
    return (
        df.groupby(['name', 'season'])[col]
        .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )


df['form']             = rolling4('total_points')
df['bonus_rate']       = rolling4('bonus')
df['xg_rolling']       = rolling4('expected_goals')
df['xa_rolling']       = rolling4('expected_assists')
df['xgc_rolling']      = rolling4('expected_goals_conceded')
df['saves_rolling']    = rolling4('saves')
df['clean_sheet_rate'] = rolling4('clean_sheets')
df['goals_rolling']    = rolling4('goals_scored')
df['assists_rolling']  = rolling4('assists')
df['mins_rolling']     = rolling4('minutes')

safe_p90 = (df['mins_rolling'] / 90).clip(lower=0.1)
df['xg_per90_r']    = (df['xg_rolling']   / safe_p90).clip(0, 5)
df['xa_per90_r']    = (df['xa_rolling']   / safe_p90).clip(0, 3)
df['xgc_per90_r']   = (df['xgc_rolling'] / safe_p90).clip(0, 5)
df['saves_per90_r'] = (df['saves_rolling']/ safe_p90).clip(0, 15)

df['target'] = df.groupby(['name', 'season'])['total_points'].shift(-1)
df = df.dropna(subset=['target', 'form']).reset_index(drop=True)
df['target'] = df['target'].astype(float)

print(f"Lignes après feature engineering : {len(df):,}\n")

# ============================================================
# SECTION 4 — Split temporel
# ============================================================
print("=" * 52)
print("  SECTION 4 — SPLIT TEMPOREL")
print("=" * 52)

train_mask = (
    (df['season'] == '2023-24') |
    ((df['season'] == '2024-25') & (df['GW'] <= 25))
)
val_mask  = (df['season'] == '2024-25') & df['GW'].between(26, 32)
test_mask = (df['season'] == '2024-25') & (df['GW'] >= 33)

df_train = df[train_mask].reset_index(drop=True)
df_val   = df[val_mask].reset_index(drop=True)
df_test  = df[test_mask].reset_index(drop=True)

print(f"TRAIN : {len(df_train):,}  |  VAL : {len(df_val):,}  |  TEST : {len(df_test):,}\n")

# ============================================================
# SECTION 5 — XGBoost par position — hyperparamètres fixes
# ============================================================
print("=" * 52)
print("  SECTION 5 — XGBOOST PAR POSITION (V3)")
print("=" * 52)

# Hyperparamètres convergés dans V1/V2 — fit direct, zéro recherche
XGB_PARAMS = {
    'n_estimators'    : 200,
    'max_depth'       : 4,
    'learning_rate'   : 0.05,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'n_jobs'          : -1,
    'random_state'    : 42,
    'verbosity'       : 0,
}

# Nouvelles features communes à toutes les positions
NEW_COMMON = [
    'form_trend', 'pts_rolling_3', 'mins_rolling_3',
    'is_endseason', 'opponent_goals_conceded_avg', 'is_top6_opponent',
]

CORE = [
    'minutes', 'mins_rolling', 'form',
    'total_points', 'bps', 'ict_index', 'value',
    'bonus_rate', 'opponent_team', 'was_home',
]
FEATURES_BY_POS = {
    'GK': CORE + NEW_COMMON + [
        'saves_rolling', 'saves_per90_r',
        'clean_sheet_rate', 'xgc_rolling', 'xgc_per90_r',
        'q_saves_per90', 'q_cs_per90', 'q_xgc_per90',
        'opponent_gc_avg_def',  # poids double pour GK
    ],
    'DEF': CORE + NEW_COMMON + [
        'influence', 'threat',
        'clean_sheet_rate', 'xgc_rolling', 'xgc_per90_r',
        'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'q_xgc_per90', 'q_cs_per90', 'q_xg_per90', 'q_xa_per90',
        'opponent_gc_avg_def',  # poids double pour DEF
    ],
    'MID': CORE + NEW_COMMON + [
        'influence', 'creativity', 'threat',
        'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'xG_rolling_3', 'xG_rolling_5',
        'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    ],
    'FWD': CORE + NEW_COMMON + [
        'influence', 'creativity', 'threat',
        'goals_rolling', 'assists_rolling',
        'xg_rolling', 'xa_rolling', 'xg_per90_r', 'xa_per90_r',
        'xG_rolling_3', 'xG_rolling_5',
        'q_xg_per90', 'q_xa_per90',
    ],
}

all_preds   = []
models_info = {}

for pos in ['GK', 'DEF', 'MID', 'FWD']:
    feats = [f for f in FEATURES_BY_POS[pos] if f in df_train.columns]

    d_tr  = df_train[df_train['position'] == pos].reset_index(drop=True)
    d_val = df_val[df_val['position']     == pos].reset_index(drop=True)
    d_te  = df_test[df_test['position']   == pos].reset_index(drop=True)

    if len(d_tr) < 30:
        print(f"  [{pos}] données insuffisantes — ignoré.")
        continue

    X_tr   = d_tr[feats].fillna(0).values;  y_tr   = d_tr['target'].values
    X_val_ = d_val[feats].fillna(0).values; y_val_ = d_val['target'].values
    X_te   = d_te[feats].fillna(0).values

    cache_file = f'model_{pos}_v3.pkl'

    if os.path.exists(cache_file):
        cached = pickle.load(open(cache_file, 'rb'))
        model  = cached['model']
        print(f"  [{pos}] ✅ modèle V3 chargé depuis cache")
    else:
        print(f"  [{pos}] {len(d_tr):>5,} train · {len(d_te):>4,} test · "
              f"{len(feats)} features — fit direct ...", end=' ', flush=True)
        model = XGBRegressor(**XGB_PARAMS)
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val_, y_val_)],
                early_stopping_rounds=20,
                verbose=False,
            )
        except TypeError:
            model.fit(X_tr, y_tr)
        pickle.dump({'model': model, 'params': XGB_PARAMS}, open(cache_file, 'wb'))
        n_trees = getattr(model, 'best_ntree_limit',
                          getattr(model, 'best_iteration', XGB_PARAMS['n_estimators']) + 1)
        print(f"OK  trees={n_trees}")

    preds = model.predict(X_te)
    d_te  = d_te.copy()
    d_te['y_pred'] = preds
    all_preds.append(d_te[['GW', 'name', 'position', 'target', 'y_pred']])

    mae_p   = mean_absolute_error(d_te['target'], preds)
    n_trees = getattr(model, 'best_ntree_limit',
                      getattr(model, 'best_iteration', XGB_PARAMS['n_estimators']) + 1)
    models_info[pos] = {'features': feats, 'n_trees': n_trees, 'params': XGB_PARAMS}
    print(f"  [{pos}] MAE={mae_p:.2f}  trees={n_trees}  feats={len(feats)}")

print()

# ============================================================
# SECTION 6 — Métriques globales TEST
# ============================================================
print("=" * 52)
print("  SECTION 6 — ÉVALUATION GLOBALE TEST")
print("=" * 52)

df_res     = pd.concat(all_preds, ignore_index=True)
y_test_all = df_res['target'].values
y_pred_all = df_res['y_pred'].values

mae  = mean_absolute_error(y_test_all, y_pred_all)
rmse = float(np.sqrt(mean_squared_error(y_test_all, y_pred_all)))
r2   = r2_score(y_test_all, y_pred_all)


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


top10_acc = top10_accuracy(df_res)

print(f"MAE        : {mae:.4f} points")
print(f"RMSE       : {rmse:.4f} points")
print(f"R²         : {r2:.4f}")
print(f"Top-10 Acc : {top10_acc:.1f}%\n")

# ============================================================
# SECTION 7 — Rapport visuel v3
# ============================================================
print("Génération rapport visuel v3 ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f'Rapport Évaluation v3 — XGBoost × 4 positions + features V3\n'
    f'MAE={mae:.2f} · RMSE={rmse:.2f} · R²={r2:.2f} · Top-10={top10_acc:.0f}%',
    fontsize=13, fontweight='bold',
)

errors = y_test_all - y_pred_all

ax1 = axes[0, 0]
pos_colors = {'GK': '#636EFA', 'DEF': '#00CC96', 'MID': '#EF553B', 'FWD': '#FFA15A'}
for pos, col in pos_colors.items():
    m = df_res['position'] == pos
    ax1.scatter(df_res.loc[m, 'target'], df_res.loc[m, 'y_pred'],
                alpha=0.25, s=6, color=col, label=pos, rasterized=True)
d_min = float(min(y_test_all.min(), y_pred_all.min()))
d_max = float(max(y_test_all.max(), y_pred_all.max()))
ax1.plot([d_min, d_max], [d_min, d_max], 'r-', lw=1.5, label='Parfait')
ax1.set_xlabel('Points réels'); ax1.set_ylabel('Points prédits')
ax1.set_title('Réels vs Prédits (par position)')
ax1.legend(fontsize=7, markerscale=2); ax1.grid(alpha=0.2)

ax2 = axes[0, 1]
ax2.hist(errors, bins=60, color='coral', edgecolor='white', alpha=0.85)
ax2.axvline(0, color='black', ls='--', lw=1.2, label='0')
ax2.axvline(errors.mean(), color='navy', ls=':', lw=1.2, label=f'μ={errors.mean():.2f}')
ax2.set_xlabel('Erreur (réels − prédits)'); ax2.set_ylabel('Fréquence')
ax2.set_title(f'Distribution des erreurs (MAE={mae:.2f})')
ax2.legend(fontsize=8)

ax3 = axes[1, 0]
mid_info  = models_info.get('MID', {})
mid_feats = mid_info.get('features', [])
if mid_feats:
    model_mid_imp = XGBRegressor(**mid_info['params'])
    X_mid_all = df_train[df_train['position'] == 'MID'][mid_feats].fillna(0).values
    y_mid_all = df_train[df_train['position'] == 'MID']['target'].values
    model_mid_imp.fit(X_mid_all, y_mid_all)
    imp      = pd.Series(model_mid_imp.feature_importances_, index=mid_feats).sort_values()
    cols_bar = ['#2ecc71' if v >= imp.median() else '#95a5a6' for v in imp.values]
    imp.plot(kind='barh', ax=ax3, color=cols_bar)
ax3.set_xlabel('Importance'); ax3.set_title('Feature Importance — modèle MID (V3)')
ax3.grid(axis='x', alpha=0.3)

ax4 = axes[1, 1]
gw_stats = (
    df_res.groupby('GW')
    .agg(pts_reels=('target', 'mean'), pts_predits=('y_pred', 'mean'))
    .reset_index()
)
ax4.plot(gw_stats['GW'], gw_stats['pts_reels'],   'b-o',  ms=5, lw=1.5, label='Réels')
ax4.plot(gw_stats['GW'], gw_stats['pts_predits'],  'r--s', ms=5, lw=1.5, label='Prédits')
ax4.fill_between(gw_stats['GW'], gw_stats['pts_reels'], gw_stats['pts_predits'],
                 alpha=0.12, color='purple')
ax4.set_xlabel('Gameweek'); ax4.set_ylabel('Points moyens')
ax4.set_title('Prédictions vs Réalité par GW (test set)')
ax4.legend(fontsize=9); ax4.grid(alpha=0.25)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_report_v3.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Rapport : {out_path}\n")

# ============================================================
# SECTION 8 — Comparaison TOUTES VERSIONS
# ============================================================
if mae < 2.5 and r2 > 0.5:
    verdict = "BON"
elif mae < 3.5 and r2 > 0.3:
    verdict = "MOYEN"
else:
    verdict = "À AMÉLIORER"

W = 57
print("=" * W)
print("        COMPARAISON TOUTES VERSIONS")
print("=" * W)
print(f"{'':22s}  {'BASELINE':>9}  {'V1/V2':>9}  {'V3(fixes)':>9}")
print("-" * W)
print(f"{'MAE':22s}  {'1.03 pts':>9}  {V1V2['MAE']:.2f} pts  {mae:.2f} pts")
print(f"{'RMSE':22s}  {'2.01 pts':>9}  {V1V2['RMSE']:.2f} pts  {rmse:.2f} pts")
print(f"{'R²':22s}  {'0.31':>9}  {V1V2['R2']:.2f}       {r2:.2f}")
v1v2_top10 = V1V2['Top10']
print(f"{'Top-10 Acc':22s}  {'10%':>9}  {v1v2_top10:.0f}%         {top10_acc:.0f}%")
print(f"{'Nouvelles features':22s}  {'Non':>9}  {'Non':>9}  {'Oui':>9}")
print(f"{'Rolling 3/5 GWs':22s}  {'Non':>9}  {'Non':>9}  {'Oui':>9}")
print("-" * W)
print(f"{'Verdict':22s}  {'MOYEN':>9}  {'À AMÉL.':>9}  {verdict:>9}")
print("=" * W)
print(f"\nRapport visuel : {out_path}")
