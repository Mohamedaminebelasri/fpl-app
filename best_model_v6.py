"""
best_model_v6.py  --  XGBoost V6
Extends V5 with:
  A1. Double GW + chance of playing (FPL API)
  A2. Bookmaker odds Bet365 (football-data.co.uk)
  A3. Elo ratings (clubelo.com)
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

print("=" * 68)
print("   BEST_MODEL_V6.PY  --  Odds + Elo + Double-GW")
print("=" * 68)
t_global = time.time()

# ---- CONSTANTS -------------------------------------------------------------
SEASONS = ['2016-17','2017-18','2018-19','2019-20',
           '2020-21','2021-22','2022-23','2023-24','2024-25']

POSITIONS    = ['GK', 'DEF', 'MID', 'FWD']
FIXED_PARAMS = dict(n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')

WALK_FOLDS = [
    ('2023-24',  1, 10),
    ('2023-24', 11, 20),
    ('2023-24', 21, 38),
    ('2024-25',  1, 19),
    ('2024-25', 20, 38),
]

# football-data.co.uk season codes
FD_SEASON_CODES = {
    '2016-17': '1617', '2017-18': '1718', '2018-19': '1819',
    '2019-20': '1920', '2020-21': '2021', '2021-22': '2122',
    '2022-23': '2223', '2023-24': '2324', '2024-25': '2425',
}

# Normalize team names to canonical lowercase key
_NM = {
    "man utd": "man utd",        "man united": "man utd",
    "manchester united": "man utd",
    "man city": "man city",      "manchester city": "man city",
    "nott'm forest": "nott'm forest",
    "nottm forest": "nott'm forest",
    "nottingham forest": "nott'm forest",
    "sheffield utd": "sheffield united",
    "sheffield united": "sheffield united",
    "aston villa": "aston villa",
    "west ham": "west ham",      "west ham united": "west ham",
    "wolves": "wolves",
    "wolverhampton wanderers": "wolves",
    "brighton": "brighton",      "brighton & hove albion": "brighton",
    "crystal palace": "crystal palace",
    "leeds": "leeds",            "leeds united": "leeds",
    "leicester": "leicester",    "leicester city": "leicester",
    "newcastle": "newcastle",    "newcastle united": "newcastle",
    "norwich": "norwich",        "norwich city": "norwich",
    "luton": "luton",            "luton town": "luton",
    "ipswich": "ipswich",        "ipswich town": "ipswich",
    "west brom": "west brom",    "west bromwich albion": "west brom",
    "huddersfield": "huddersfield",
    "cardiff": "cardiff",        "cardiff city": "cardiff",
    "swansea": "swansea",        "swansea city": "swansea",
    # clubelo lowercase keys
    "manunited": "man utd",      "mancity": "man city",
    "astonvilla": "aston villa", "westham": "west ham",
    "nottmforest": "nott'm forest",
    "crystalpalace": "crystal palace",
    "sheffieldunited": "sheffield united",
    "westbrom": "west brom",
}
def norm_team(t) -> str:
    s = str(t).strip().lower()
    return _NM.get(s, s)

# Canonical key -> clubelo API slug
_TEAM_TO_ELO = {
    "arsenal": "Arsenal",            "aston villa": "AstonVilla",
    "bournemouth": "Bournemouth",    "brentford": "Brentford",
    "brighton": "Brighton",          "burnley": "Burnley",
    "cardiff": "Cardiff",            "chelsea": "Chelsea",
    "crystal palace": "CrystalPalace",
    "everton": "Everton",            "fulham": "Fulham",
    "huddersfield": "Huddersfield",  "ipswich": "Ipswich",
    "leeds": "Leeds",                "leicester": "Leicester",
    "liverpool": "Liverpool",        "luton": "Luton",
    "man city": "ManCity",           "man utd": "ManUnited",
    "newcastle": "Newcastle",        "nott'm forest": "NottmForest",
    "norwich": "Norwich",            "sheffield united": "SheffieldUnited",
    "southampton": "Southampton",    "swansea": "Swansea",
    "tottenham": "Tottenham",        "watford": "Watford",
    "west brom": "WestBrom",         "west ham": "WestHam",
    "wolves": "Wolves",
}

# ---- HELPERS ---------------------------------------------------------------
def top10_accuracy(df_ev: pd.DataFrame) -> float:
    hits, n_gw = 0, 0
    for gw in sorted(df_ev['GW'].unique()):
        g = df_ev[df_ev['GW'] == gw]
        if len(g) < 10: continue
        hits += len(
            set(g.nlargest(10, 'y_pred').index) &
            set(g.nlargest(10, 'target').index)
        )
        n_gw += 1
    return (hits / (n_gw * 10)) * 100 if n_gw > 0 else 0.0

def evaluate(yt, yp):
    return (mean_absolute_error(yt, yp),
            mean_squared_error(yt, yp) ** 0.5,
            r2_score(yt, yp))

def load_optuna(pos: str) -> dict:
    p = f'optuna_XGB_{pos}.pkl'
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    return dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2)

def train_model(df_tr, pos, features, optuna_params, sw=1.5):
    d = df_tr[df_tr['position'] == pos]
    if len(d) < 50: return None
    X = d[features].fillna(0).values
    y = d['target'].values
    w = np.where(y >= 8, sw, 1.0)
    mdl = XGBRegressor(**{**optuna_params, **FIXED_PARAMS})
    mdl.fit(X, y, sample_weight=w, verbose=False)
    return mdl

def predict_all(models, df_te, features):
    preds = pd.Series(np.nan, index=df_te.index)
    for pos, mdl in models.items():
        if mdl is None: continue
        mask = df_te['position'] == pos
        if mask.sum() == 0: continue
        preds.loc[mask] = mdl.predict(df_te.loc[mask, features].fillna(0).values)
    return preds

HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (FPL-Model-V6; academic)'}


# =============================================================================
# SECTION 1  --  Load V5 base data
# =============================================================================
print("\n[1] Chargement des donnees V5 (cache_9seasons_features.pkl) ...")

if not os.path.exists('cache_9seasons_features.pkl'):
    print("   ERREUR: cache_9seasons_features.pkl manquant -> lance best_model_v5.py d'abord")
    sys.exit(1)

df_base = pickle.load(open('cache_9seasons_features.pkl', 'rb'))
df_base['team_norm']     = df_base['team'].apply(norm_team)
df_base['kickoff_date']  = pd.to_datetime(
    df_base.get('kickoff_time', pd.NaT), errors='coerce', utc=True
).dt.date

# Load V5 top-20 features as starting point
if os.path.exists('shap_top_features.pkl'):
    v5_top20 = pickle.load(open('shap_top_features.pkl', 'rb'))
else:
    v5_top20 = []

print(f"   Base: {len(df_base):,} lignes | {df_base.shape[1]} colonnes")
print(f"   V5 top-20: {v5_top20}")


# =============================================================================
# AMELIORATION 1  --  Double GW + Chance of Playing
# =============================================================================
print("\n[A1] Double GW + Chance of Playing ...")
CACHE_AVAIL = 'cache_availability.pkl'

if os.path.exists(CACHE_AVAIL):
    print(f"   Cache '{CACHE_AVAIL}' trouve.")
    avail_df = pickle.load(open(CACHE_AVAIL, 'rb'))
else:
    avail_df = df_base[['name', 'season', 'GW', 'minutes', 'mins_rolling_3',
                         'team', 'team_norm']].copy()

    # -- DGW heuristic from minutes ------------------------------------
    avail_df['games_this_gw'] = (1 + (avail_df['minutes'] > 90).astype(int)).astype(float)
    avail_df['is_double_gw']  = (avail_df['minutes'] > 90).astype(int)
    avail_df['is_blank_gw']   = 0  # hard to detect reliably from data alone

    # -- chance_of_playing proxy from rolling minutes ------------------
    avail_df['chance_of_playing'] = (avail_df['mins_rolling_3'] / 90.0).clip(0, 1)

    # -- Supplement 2024-25 with FPL bootstrap-static API -------------
    try:
        print("   FPL bootstrap-static API ...")
        resp = requests.get(
            'https://fantasy.premierleague.com/api/bootstrap-static/',
            timeout=15, headers=HTTP_HEADERS
        )
        bs = resp.json()
        # team id -> name
        fpl_teams = {t['id']: t['name'] for t in bs.get('teams', [])}
        # player cop map: web_name + full_name -> float 0-1
        cop_map = {}
        for e in bs.get('elements', []):
            cop = e.get('chance_of_playing_next_round')
            val = 1.0 if cop is None else cop / 100.0
            wn = str(e.get('web_name', '')).strip().lower()
            fn = f"{e.get('first_name','')} {e.get('second_name','')}".strip().lower()
            cop_map[wn] = val
            if fn: cop_map[fn] = val

        # apply to 2024-25 rows
        mask_cur = avail_df['season'] == '2024-25'
        avail_df.loc[mask_cur, 'chance_of_playing'] = (
            avail_df.loc[mask_cur, 'name']
            .str.lower()
            .map(cop_map)
            .fillna(avail_df.loc[mask_cur, 'chance_of_playing'])
        )
        print(f"   cop_map: {len(cop_map)} joueurs depuis API")

        # DGW detection for 2024-25 from fixtures API
        resp2 = requests.get(
            'https://fantasy.premierleague.com/api/fixtures/',
            timeout=15, headers=HTTP_HEADERS
        )
        fixtures = resp2.json()
        gw_team_cnt = {}
        for f in fixtures:
            gw = f.get('event')
            if gw is None: continue
            for tid in [f['team_h'], f['team_a']]:
                gw_team_cnt[(gw, tid)] = gw_team_cnt.get((gw, tid), 0) + 1
        # map (gw, tid) -> team_name_norm -> is_dgw
        dgw_lookup = {}
        for (gw, tid), cnt in gw_team_cnt.items():
            tname = norm_team(fpl_teams.get(tid, str(tid)))
            dgw_lookup[(gw, tname)] = cnt
        # apply to 2024-25
        def get_games_api(row):
            return float(dgw_lookup.get((row['GW'], row['team_norm']), 1.0))
        avail_df.loc[mask_cur, 'games_this_gw'] = (
            avail_df.loc[mask_cur].apply(get_games_api, axis=1)
        )
        avail_df.loc[mask_cur, 'is_double_gw'] = (
            avail_df.loc[mask_cur, 'games_this_gw'] >= 2
        ).astype(int)
        print(f"   DGW detected for 2024-25: {int(avail_df.loc[mask_cur,'is_double_gw'].sum())} rows")
    except Exception as e:
        print(f"   FPL API error (continuing): {e}")

    # -- expected_minutes -----------------------------------------------
    avail_df['expected_minutes'] = (
        avail_df['mins_rolling_3']
        * avail_df['chance_of_playing']
        * avail_df['games_this_gw'].clip(1, 2)
    )

    pickle.dump(avail_df, open(CACHE_AVAIL, 'wb'))
    print(f"   Sauvegarde '{CACHE_AVAIL}'")

avail_cols = ['name', 'season', 'GW', 'games_this_gw', 'is_double_gw',
              'is_blank_gw', 'chance_of_playing', 'expected_minutes']
print(f"   DGW rows: {avail_df['is_double_gw'].sum():,} | "
      f"avg chance_of_playing: {avail_df['chance_of_playing'].mean():.2f}")


# =============================================================================
# AMELIORATION 2  --  Bookmaker Odds (football-data.co.uk)
# =============================================================================
print("\n[A2] Odds Bet365 (football-data.co.uk) ...")
CACHE_ODDS = 'cache_odds.pkl'

if os.path.exists(CACHE_ODDS):
    print(f"   Cache '{CACHE_ODDS}' trouve.")
    odds_df = pickle.load(open(CACHE_ODDS, 'rb'))
else:
    frames_odds = []
    for season, code in FD_SEASON_CODES.items():
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        try:
            resp = requests.get(url, timeout=30, headers=HTTP_HEADERS)
            resp.raise_for_status()
            # football-data CSVs sometimes have BOM
            text = resp.text.lstrip('﻿')
            df_o = pd.read_csv(io.StringIO(text), on_bad_lines='skip',
                               encoding='utf-8', low_memory=False)
            df_o['season'] = season

            # Rename BOM-prefixed Div column if present
            df_o.columns = [c.lstrip('﻿') for c in df_o.columns]

            # Parse date (try DD/MM/YY and DD/MM/YYYY)
            df_o['match_date'] = pd.to_datetime(
                df_o['Date'], dayfirst=True, errors='coerce'
            ).dt.date

            # Required cols
            for c in ['B365H','B365D','B365A']:
                df_o[c] = pd.to_numeric(df_o.get(c, np.nan), errors='coerce')

            df_o['home_team_norm'] = df_o['HomeTeam'].apply(norm_team)
            df_o['away_team_norm'] = df_o['AwayTeam'].apply(norm_team)

            # Drop rows without valid odds or date
            df_o = df_o.dropna(subset=['B365H','B365D','B365A','match_date'])
            if len(df_o) == 0:
                continue

            # Implied probabilities (overround-normalised)
            inv_h = 1 / df_o['B365H']
            inv_d = 1 / df_o['B365D']
            inv_a = 1 / df_o['B365A']
            total = inv_h + inv_d + inv_a
            df_o['prob_home_win'] = (inv_h / total).clip(0, 1)
            df_o['prob_draw']     = (inv_d / total).clip(0, 1)
            df_o['prob_away_win'] = (inv_a / total).clip(0, 1)

            # Over 2.5
            if 'B365>2.5' in df_o.columns:
                df_o['B365_over25'] = pd.to_numeric(df_o['B365>2.5'], errors='coerce')
                df_o['prob_over25'] = (1 / df_o['B365_over25']).clip(0, 1)
            else:
                # estimate: high-scoring matches more likely if both teams strong
                df_o['prob_over25'] = (
                    (df_o['prob_home_win'] + df_o['prob_away_win']) * 0.72
                ).clip(0, 0.85)

            df_o['prob_clean_sheet_approx'] = (
                1 - df_o['prob_over25'] * 0.6
            ).clip(0, 1)

            frames_odds.append(df_o[[
                'season', 'match_date', 'home_team_norm', 'away_team_norm',
                'prob_home_win', 'prob_away_win', 'prob_draw',
                'prob_over25', 'prob_clean_sheet_approx',
            ]])
            print(f"   {season}: {len(df_o):>3} matches OK")
        except Exception as e:
            print(f"   {season}: ERREUR ({e})")

    if frames_odds:
        odds_df = pd.concat(frames_odds, ignore_index=True)
        pickle.dump(odds_df, open(CACHE_ODDS, 'wb'))
        print(f"   Sauvegarde '{CACHE_ODDS}': {len(odds_df):,} matches")
    else:
        odds_df = pd.DataFrame()
        print("   Aucune donnee odds recuperee")

print(f"   Odds: {len(odds_df):,} matches sur {len(odds_df['season'].unique()) if len(odds_df)>0 else 0} saisons")


# =============================================================================
# AMELIORATION 3  --  Elo ratings (clubelo.com)
# =============================================================================
print("\n[A3] Elo ratings (clubelo.com) ...")
CACHE_ELO = 'cache_elo.pkl'

if os.path.exists(CACHE_ELO):
    print(f"   Cache '{CACHE_ELO}' trouve.")
    elo_ratings = pickle.load(open(CACHE_ELO, 'rb'))
else:
    elo_ratings = {}  # {norm_team_key: (sorted_dates_np, sorted_elos_np)}
    teams_ok, teams_fail = 0, 0

    for canon_key, elo_slug in _TEAM_TO_ELO.items():
        url = f"http://api.clubelo.com/{elo_slug}"
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=15, headers=HTTP_HEADERS)
                resp.raise_for_status()
                df_e = pd.read_csv(io.StringIO(resp.text), low_memory=False)
                # clubelo format: Rank,Club,Country,Level,Elo,From,To
                if 'Elo' not in df_e.columns or 'From' not in df_e.columns:
                    break
                df_e['date'] = pd.to_datetime(df_e['From'], errors='coerce')
                df_e = df_e.dropna(subset=['date', 'Elo'])
                # Filter 2015-2026
                df_e = df_e[(df_e['date'].dt.year >= 2015) &
                             (df_e['date'].dt.year <= 2026)]
                df_e = df_e.sort_values('date')
                dates_arr = df_e['date'].values.astype('datetime64[D]')
                elos_arr  = df_e['Elo'].values.astype(float)
                elo_ratings[canon_key] = (dates_arr, elos_arr)
                teams_ok += 1
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    teams_fail += 1
                    print(f"   {elo_slug}: ECHEC ({e})")

    pickle.dump(elo_ratings, open(CACHE_ELO, 'wb'))
    print(f"   Sauvegarde '{CACHE_ELO}': {teams_ok} equipes OK, {teams_fail} echecs")

def get_elo(team_key: str, match_date, default: float = 1500.0) -> float:
    key = norm_team(team_key)
    if key not in elo_ratings or match_date is None:
        return default
    try:
        dates_arr, elos_arr = elo_ratings[key]
        d64 = np.datetime64(str(match_date), 'D')
        idx = int(np.searchsorted(dates_arr, d64, side='right')) - 1
        if idx < 0:
            return float(elos_arr[0]) if len(elos_arr) > 0 else default
        return float(elos_arr[min(idx, len(elos_arr) - 1)])
    except Exception:
        return default

print(f"   Elo disponible pour {len(elo_ratings)} equipes: {list(elo_ratings.keys())[:5]} ...")


# =============================================================================
# SECTION 4  --  Feature Fusion
# =============================================================================
print("\n[4] Fusion des features ...")
CACHE_V6 = 'cache_v6_features.pkl'

if os.path.exists(CACHE_V6):
    print(f"   Cache '{CACHE_V6}' trouve.")
    df_v6 = pickle.load(open(CACHE_V6, 'rb'))
else:
    df = df_base.copy()

    # -- A1: Availability --------------------------------------------------
    print("   Merge disponibilite ...")
    df = df.merge(
        avail_df[avail_cols],
        on=['name', 'season', 'GW'],
        how='left'
    )
    for c in ['games_this_gw', 'is_double_gw', 'is_blank_gw',
              'chance_of_playing', 'expected_minutes']:
        df[c] = df[c].fillna(0 if c != 'chance_of_playing' else 1.0)

    # -- A2: Odds ----------------------------------------------------------
    print("   Merge odds ...")
    if len(odds_df) > 0:
        df['kickoff_date'] = pd.to_datetime(
            df.get('kickoff_time', pd.NaT), errors='coerce', utc=True
        ).dt.date

        ODDS_COLS = ['prob_team_win', 'prob_draw', 'prob_over25',
                     'prob_clean_sheet_approx']
        for c in ODDS_COLS:
            df[c] = np.nan

        # Prepare home odds
        home_odds = odds_df[['season', 'match_date', 'home_team_norm',
                              'prob_home_win', 'prob_draw', 'prob_over25',
                              'prob_clean_sheet_approx']].copy()
        home_odds = home_odds.rename(columns={
            'home_team_norm': '_team_key',
            'prob_home_win': '_prob_win'
        })
        home_odds['_is_home'] = 1

        # Prepare away odds
        away_odds = odds_df[['season', 'match_date', 'away_team_norm',
                              'prob_away_win', 'prob_draw', 'prob_over25',
                              'prob_clean_sheet_approx']].copy()
        away_odds = away_odds.rename(columns={
            'away_team_norm': '_team_key',
            'prob_away_win': '_prob_win'
        })
        away_odds['_is_home'] = 0

        all_odds = pd.concat([home_odds, away_odds], ignore_index=True)
        all_odds['match_date'] = pd.to_datetime(
            all_odds['match_date'], errors='coerce'
        ).dt.date

        # For each row in df, match on (season, date, team_norm, was_home)
        df_home_mask = df['was_home'] == 1
        df_away_mask = df['was_home'] == 0

        for mask, is_home in [(df_home_mask, 1), (df_away_mask, 0)]:
            sub = df.loc[mask, ['season', 'kickoff_date', 'team_norm']].copy()
            sub.columns = ['season', 'match_date', '_team_key']
            sub['_is_home'] = is_home
            sub['_orig_idx'] = df.loc[mask].index

            merged = sub.merge(
                all_odds[all_odds['_is_home'] == is_home],
                on=['season', 'match_date', '_team_key', '_is_home'],
                how='left'
            )
            merged = merged.drop_duplicates(subset=['_orig_idx'], keep='first')
            merged = merged.set_index('_orig_idx')

            for col, src in [('prob_team_win', '_prob_win'),
                              ('prob_draw', 'prob_draw'),
                              ('prob_over25', 'prob_over25'),
                              ('prob_clean_sheet_approx', 'prob_clean_sheet_approx')]:
                if src in merged.columns:
                    df.loc[merged.index, col] = merged[src].values

        for c in ODDS_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        df['odds_goals_context'] = df['prob_over25'] * df['games_this_gw'].clip(1, 2)
        n_matched = (df['prob_team_win'] > 0).sum()
        print(f"   Odds: {n_matched:,} lignes matchees ({n_matched/len(df)*100:.1f}%)")
    else:
        for c in ['prob_team_win', 'prob_draw', 'prob_over25',
                  'prob_clean_sheet_approx', 'odds_goals_context']:
            df[c] = 0.0
        print("   Odds: aucune donnee, features = 0")

    # -- A3: Elo -----------------------------------------------------------
    print("   Calcul Elo (join par date) ...")
    elo_features = ['elo_team', 'elo_opponent', 'elo_diff', 'elo_win_prob']
    for c in elo_features:
        df[c] = np.nan

    if elo_ratings:
        # Only process rows where team_norm is in elo_ratings
        # AND kickoff_date is not null
        df['kickoff_date'] = pd.to_datetime(
            df.get('kickoff_time', pd.NaT), errors='coerce', utc=True
        ).dt.date

        # Batch computation for speed
        unique_teams = df['team_norm'].unique()
        team_in_elo  = set(elo_ratings.keys())

        # Vectorized Elo lookup per team
        for team_key in unique_teams:
            mask = df['team_norm'] == team_key
            if mask.sum() == 0: continue
            sub = df.loc[mask, ['kickoff_date']].copy()
            # Get Elo for each unique date
            unique_dates = sub['kickoff_date'].dropna().unique()
            date_to_elo = {d: get_elo(team_key, d) for d in unique_dates}
            df.loc[mask, 'elo_team'] = sub['kickoff_date'].map(date_to_elo)

        # Opponent Elo: opponent_team column contains team ID
        # For 2020+ we have opponent canonical name from opponent_goals_conceded_avg
        # Workaround: use team_norm of opponent via fixture pairing
        # For simplicity: use opponent_team as key if it maps to a canonical name
        df['opp_team_norm'] = df['opponent_team'].astype(str).apply(norm_team)
        for opp_key in df['opp_team_norm'].unique():
            mask = df['opp_team_norm'] == opp_key
            if mask.sum() == 0: continue
            sub = df.loc[mask, ['kickoff_date']].copy()
            unique_dates = sub['kickoff_date'].dropna().unique()
            date_to_elo = {d: get_elo(opp_key, d) for d in unique_dates}
            df.loc[mask, 'elo_opponent'] = sub['kickoff_date'].map(date_to_elo)

        # Fill defaults and compute derived features
        df['elo_team']     = pd.to_numeric(df['elo_team'],     errors='coerce').fillna(1500)
        df['elo_opponent'] = pd.to_numeric(df['elo_opponent'], errors='coerce').fillna(1500)
        df['elo_diff']     = df['elo_team'] - df['elo_opponent']
        df['elo_win_prob'] = 1 / (1 + 10 ** (-df['elo_diff'] / 400))

        n_elo = (df['elo_team'] != 1500).sum()
        print(f"   Elo: {n_elo:,} lignes avec rating non-defaut ({n_elo/len(df)*100:.1f}%)")
    else:
        for c in elo_features:
            df[c] = 0.0
        print("   Elo: aucune donnee, features = 0")

    df_v6 = df.reset_index(drop=True)
    pickle.dump(df_v6, open(CACHE_V6, 'wb'))
    print(f"   Sauvegarde '{CACHE_V6}': {len(df_v6):,} lignes, {df_v6.shape[1]} colonnes")

print(f"   Shape V6: {df_v6.shape}")

# New features from A1/A2/A3
NEW_V6_FEATURES = [
    'games_this_gw', 'is_double_gw', 'chance_of_playing', 'expected_minutes',
    'prob_team_win', 'prob_draw', 'prob_over25',
    'prob_clean_sheet_approx', 'odds_goals_context',
    'elo_team', 'elo_opponent', 'elo_diff', 'elo_win_prob',
]


# =============================================================================
# SECTION 5  --  SHAP TOP-25 + Retrain
# =============================================================================
print("\n[5] SHAP TOP-25 + Reentrainement ...")
CACHE_SHAP_V6 = 'shap_v6_top_features.pkl'

# All candidate features = V5 top-20 + new V6 features + extras
ALL_CANDIDATES = list(dict.fromkeys(
    v5_top20 +
    NEW_V6_FEATURES + [
        'pts_rolling_1', 'pts_rolling_3', 'pts_rolling_5', 'pts_rolling_10',
        'form_momentum', 'value_score', 'consistency', 'big_game_flag',
        'team_avg_pts_scored', 'team_clean_sheet_rate',
        'team_goals_per_game', 'opponent_goals_conceded_avg',
        'is_endseason', 'is_earlyseason', 'gw_normalized', 'season_order',
        'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
        'q_xgc_per90', 'q_cs_per90', 'q_saves_per90',
        'bps_rolling_3', 'goals_rolling_5', 'assists_rolling_5',
        'xg_rolling_3', 'xg_rolling_5', 'xa_rolling_3',
    ]
))
ALL_CANDIDATES = [f for f in ALL_CANDIDATES if f in df_v6.columns]
print(f"   Candidats SHAP: {len(ALL_CANDIDATES)} features")

if os.path.exists(CACHE_SHAP_V6):
    print(f"   Cache '{CACHE_SHAP_V6}' trouve.")
    top25_features = pickle.load(open(CACHE_SHAP_V6, 'rb'))
else:
    mask_shap = (df_v6['season'] != '2024-25') | (df_v6['GW'] <= 28)
    df_shap_pool = df_v6[mask_shap].copy()

    np.random.seed(42)
    n_s = max(5000, int(0.20 * len(df_shap_pool)))
    idx = np.random.choice(len(df_shap_pool), size=n_s, replace=False)
    Xs = df_shap_pool[ALL_CANDIDATES].fillna(0).values[idx]
    ys = df_shap_pool['target'].values[idx]

    print(f"   SHAP: {n_s:,} samples x {len(ALL_CANDIDATES)} features ...")
    t0 = time.time()
    qm = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, **FIXED_PARAMS)
    qm.fit(Xs, ys, verbose=False)

    try:
        explainer   = shap.TreeExplainer(qm)
        shap_vals   = explainer.shap_values(Xs, check_additivity=False)
        shap_imp    = np.abs(shap_vals).mean(axis=0)
        top25_idx   = np.argsort(shap_imp)[::-1][:25]
        top25_features = [ALL_CANDIDATES[i] for i in top25_idx]
        shap_scores = shap_imp[top25_idx]
        method = 'SHAP'
    except Exception as e:
        print(f"   SHAP erreur ({e}), fallback importance")
        imp = qm.feature_importances_
        top25_idx = np.argsort(imp)[::-1][:25]
        top25_features = [ALL_CANDIDATES[i] for i in top25_idx]
        shap_scores = imp[top25_idx]
        method = 'XGB_importance'

    print(f"   Methode: {method} | Duree: {time.time()-t0:.1f}s")
    pickle.dump(top25_features, open(CACHE_SHAP_V6, 'wb'))
    print(f"   Sauvegarde '{CACHE_SHAP_V6}'")

print(f"   Top 25 features V6:")
for i, f in enumerate(top25_features, 1):
    in_v5 = '*' if f in v5_top20 else '+'
    print(f"     {i:2d}. {in_v5} {f}")
print("     (* = existait en V5  |  + = nouvelle/montee V6)")

# -- Retrain 4 XGBoost models -----------------------------------------------
print("   Reentrainement des 4 modeles XGBoost ...")

mask_train_final = (
    (df_v6['season'] != '2024-25') | (df_v6['GW'] <= 32)
)
mask_test_final = (
    (df_v6['season'] == '2024-25') & (df_v6['GW'] >= 33)
)
df_train_final = df_v6[mask_train_final].copy()
df_test_final  = df_v6[mask_test_final].copy()
print(f"   Train: {len(df_train_final):,} | Test GW33+: {len(df_test_final):,}")

final_models_v6 = {}
for pos in POSITIONS:
    op = load_optuna(pos)
    mdl = train_model(df_train_final, pos, top25_features, op)
    if mdl is not None:
        final_models_v6[pos] = mdl
        pickle.dump(mdl, open(f'model_v6_{pos}.pkl', 'wb'))
        n_p = (df_train_final['position'] == pos).sum()
        print(f"   {pos}: {n_p:,} samples -> model_v6_{pos}.pkl")


# =============================================================================
# SECTION 6  --  Walk-Forward + Rapport
# =============================================================================
print("\n[6] Walk-Forward Validation + Rapport ...")

wf_results   = []
all_preds_wf = []

for fold_i, (test_season, gw_start, gw_end) in enumerate(WALK_FOLDS, 1):
    mask_tr = (
        (df_v6['season'] != test_season) | (df_v6['GW'] < gw_start)
    )
    mask_te = (
        (df_v6['season'] == test_season) &
        (df_v6['GW'] >= gw_start) &
        (df_v6['GW'] <= gw_end)
    )
    df_tr = df_v6[mask_tr].copy()
    df_te = df_v6[mask_te].copy()
    if len(df_te) < 50: continue

    fold_models = {pos: train_model(df_tr, pos, top25_features, load_optuna(pos))
                   for pos in POSITIONS}

    df_te = df_te.copy()
    df_te['y_pred'] = predict_all(fold_models, df_te, top25_features)
    df_te_c = df_te.dropna(subset=['y_pred'])
    if len(df_te_c) == 0: continue

    mae, rmse, r2 = evaluate(df_te_c['target'], df_te_c['y_pred'])
    t10 = top10_accuracy(df_te_c.reset_index(drop=True))
    wf_results.append({'fold': fold_i, 'season': test_season,
                       'gw': f'{gw_start}-{gw_end}',
                       'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Top10': t10,
                       'n': len(df_te_c)})
    all_preds_wf.append(df_te_c)
    print(f"   Fold {fold_i} ({test_season} GW{gw_start}-{gw_end}): "
          f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  "
          f"Top-10={t10:.1f}%  (n={len(df_te_c):,})")

if wf_results:
    wf_df   = pd.DataFrame(wf_results)
    wf_mae  = wf_df['MAE'].mean()
    wf_rmse = wf_df['RMSE'].mean()
    wf_r2   = wf_df['R2'].mean()
    wf_top10= wf_df['Top10'].mean()
else:
    wf_mae = wf_rmse = wf_r2 = wf_top10 = float('nan')

print(f"\n   -- Walk-Forward Moyen V6 --")
print(f"   MAE={wf_mae:.4f}  RMSE={wf_rmse:.4f}  R2={wf_r2:.4f}  Top-10={wf_top10:.1f}%")

# Final eval GW33+
if len(df_test_final) >= 50:
    df_test_final = df_test_final.copy()
    df_test_final['y_pred'] = predict_all(final_models_v6, df_test_final, top25_features)
    df_ev = df_test_final.dropna(subset=['y_pred'])
    final_mae, final_rmse, final_r2 = evaluate(df_ev['target'], df_ev['y_pred'])
    final_top10 = top10_accuracy(df_ev.reset_index(drop=True))
elif all_preds_wf:
    df_all_wf = pd.concat(all_preds_wf, ignore_index=True)
    final_mae, final_rmse, final_r2 = evaluate(
        df_all_wf['target'], df_all_wf['y_pred'])
    final_top10 = top10_accuracy(df_all_wf)
else:
    final_mae = final_rmse = final_r2 = final_top10 = float('nan')

print(f"\n   -- Score Final V6 (GW33+) --")
print(f"   MAE={final_mae:.4f}  RMSE={final_rmse:.4f}  "
      f"R2={final_r2:.4f}  Top-10={final_top10:.1f}%")

# -- Comparison table -------------------------------------------------------
def pct_gain(new, old, higher_better=False):
    if np.isnan(new) or np.isnan(old) or old == 0: return "N/A"
    gain = (old - new) / abs(old) * 100 if not higher_better else (new - old) / abs(old) * 100
    sign = '+' if gain > 0 else ''
    return f"{sign}{gain:.1f}%"

versions = [
    {'Version': 'Baseline RF',   'Saisons': 2, 'Feat': 8,
     'MAE': 1.03, 'RMSE': 2.01, 'R2': 0.31, 'Top10': 10.0},
    {'Version': 'XGBoost V3',    'Saisons': 2, 'Feat': 38,
     'MAE': 0.95, 'RMSE': 1.86, 'R2': 0.28, 'Top10': 6.0},
    {'Version': 'Ensemble V4',   'Saisons': 2, 'Feat': 42,
     'MAE': 1.09, 'RMSE': 2.05, 'R2': 0.29, 'Top10': 8.0},
    {'Version': 'XGBoost V5 WF', 'Saisons': 9, 'Feat': 'TOP20',
     'MAE': 1.0225, 'RMSE': 1.9786, 'R2': 0.2926, 'Top10': 11.9},
    {'Version': 'XGBoost V5',    'Saisons': 9, 'Feat': 'TOP20',
     'MAE': 1.0432, 'RMSE': 2.0213, 'R2': 0.3175, 'Top10': 10.0},
    {'Version': 'XGBoost V6 WF', 'Saisons': 9, 'Feat': 'TOP25',
     'MAE': round(wf_mae,4), 'RMSE': round(wf_rmse,4),
     'R2': round(wf_r2,4),   'Top10': round(wf_top10,1)},
    {'Version': 'XGBoost V6',    'Saisons': 9, 'Feat': 'TOP25',
     'MAE': round(final_mae,4), 'RMSE': round(final_rmse,4),
     'R2': round(final_r2,4),   'Top10': round(final_top10,1)},
]

sep = "=" * 72
sub = "-" * 72
print(f"\n{sep}")
print("              EVOLUTION COMPLETE DU MODELE FPL")
print(sep)
print(f"{'Version':<16} {'Saisons':>7} {'Feat':>7} {'MAE':>7} {'RMSE':>7} {'R2':>6} {'Top-10':>8}")
print(sub)
for v in versions:
    print(f"{v['Version']:<16} {v['Saisons']:>7} {str(v['Feat']):>7} "
          f"{v['MAE']:>7.4f} {v['RMSE']:>7.4f} {v['R2']:>6.4f} {v['Top10']:>7.1f}%")
print(sub)
print("Nouvelles features V6:")
print("  + Double GW / Blank GW flag")
print("  + % Chance de jouer")
print("  + Cotes Bet365 (prob victoire, CS, buts)")
print("  + Elo equipes (force relative)")
print(sub)
print(f"Gain V6 vs V5:  "
      f"MAE {pct_gain(final_mae, 1.0432)}  "
      f"R2 {pct_gain(final_r2, 0.3175, higher_better=True)}  "
      f"Top-10 {pct_gain(final_top10, 10.0, higher_better=True)}")
print(sep)

# -- PNG Report -------------------------------------------------------------
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("XGBoost V6  --  Odds + Elo + Double-GW",
                 fontsize=14, fontweight='bold')

    ver_names = ['RF', 'V3', 'V4', 'V5', 'V6']
    mae_vals  = [1.03, 0.95, 1.09, 1.0432, final_mae]
    rmse_vals = [2.01, 1.86, 2.05, 2.0213, final_rmse]
    r2_vals   = [0.31, 0.28, 0.29, 0.3175, final_r2]
    t10_vals  = [10.0, 6.0,  8.0,  10.0,   final_top10]
    colors    = ['#aad4f0','#aad4f0','#aad4f0','#62b3e0','#1a7abf']

    # Graph 1: MAE evolution
    ax = axes[0, 0]
    bars = ax.bar(ver_names, mae_vals, color=colors)
    ax.set_title('Evolution MAE (bas = mieux)', fontweight='bold')
    ax.set_ylabel('MAE')
    for bar, v in zip(bars, mae_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, max(x for x in mae_vals if not np.isnan(x)) * 1.25)

    # Graph 2: R2 evolution
    ax = axes[0, 1]
    bars = ax.bar(ver_names, r2_vals, color=colors)
    ax.set_title('Evolution R2 (haut = mieux)', fontweight='bold')
    ax.set_ylabel('R2')
    for bar, v in zip(bars, r2_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, max(x for x in r2_vals if not np.isnan(x)) * 1.3)

    # Graph 3: Top-10 evolution
    ax = axes[1, 0]
    bars = ax.bar(ver_names, t10_vals, color=colors)
    ax.set_title('Evolution Top-10% (haut = mieux)', fontweight='bold')
    ax.set_ylabel('Top-10 Accuracy (%)')
    for bar, v in zip(bars, t10_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, max(x for x in t10_vals if not np.isnan(x)) * 1.3)

    # Graph 4: Top-25 SHAP features
    ax = axes[1, 1]
    if final_models_v6 and top25_features:
        imp_sum = np.zeros(len(top25_features))
        n_m = 0
        for pos, mdl in final_models_v6.items():
            if mdl is None: continue
            feat_nm = (list(mdl.feature_names_in_)
                       if hasattr(mdl, 'feature_names_in_') else top25_features)
            imp_map = dict(zip(feat_nm, mdl.feature_importances_))
            for i, f in enumerate(top25_features):
                imp_sum[i] += imp_map.get(f, 0)
            n_m += 1
        if n_m: imp_sum /= n_m
        sidx = np.argsort(imp_sum)
        bar_colors = ['#e07b39' if top25_features[i] in NEW_V6_FEATURES
                      else '#62b3e0' for i in sidx]
        ax.barh([top25_features[i] for i in sidx], imp_sum[sidx],
                color=bar_colors, alpha=0.85)
        ax.set_title('Top-25 importance (orange=nouveau V6)', fontweight='bold')
        ax.set_xlabel('Importance (moy. 4 pos.)')
        # legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#e07b39', label='Nouveau V6'),
            Patch(color='#62b3e0', label='Existant V5'),
        ], fontsize=7, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No models', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('best_model_v6_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n   Rapport sauvegarde: best_model_v6_report.png")
except Exception as e:
    print(f"\n   Erreur rapport PNG: {e}")

total = time.time() - t_global
print(f"\n   Duree totale: {total/60:.1f} min")
print("=" * 68)
print("   TERMINE  --  XGBoost V6")
print("=" * 68)
