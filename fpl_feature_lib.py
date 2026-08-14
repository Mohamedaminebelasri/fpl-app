"""
fpl_feature_lib.py
Logique de feature engineering partagee (extraite telle quelle de
best_model_v5.py / best_model_v6.py) pour rester 100% coherente avec les
modeles deja entraines (model_v6_*.pkl + shap_v6_top_features.pkl).

Utilisee par :
  - backfill_2025_26.py   (backfill de la saison manquante dans le cache)
  - fpl_app_final33.py    (get_xp_predictions -> features live + fallback)
  - retrain_weekly.py     (ingestion GW + retrain hebdo 2026/27)
"""
import os
import io
import pickle
import numpy as np
import pandas as pd
import requests

SEASONS_V6 = [
    '2016-17', '2017-18', '2018-19', '2019-20',
    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26',
]
SEASON_ORDER = {s: i + 1 for i, s in enumerate(SEASONS_V6)}

BASE_URL = ("https://raw.githubusercontent.com/vaastav/"
            "Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv")
PLAYERS_URL = ("https://raw.githubusercontent.com/vaastav/"
               "Fantasy-Premier-League/master/data/{season}/players_raw.csv")

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
POS_MAP = {
    1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD',
    '1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD',
    'GK': 'GK', 'GKP': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD',
    'AM': 'MID', 'Goalkeeper': 'GK',
    'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD',
    'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD', 'ATT': 'FWD',
}

QUALITY_COLS = [
    'q_xg_per90', 'q_xa_per90', 'q_xgi_per90',
    'q_xgc_per90', 'q_cs_per90', 'q_saves_per90',
]

def model_confidence(gw: int) -> int:
    """
    Retourne un score de confiance 0-100 selon la phase de saison, base sur
    la completude des rolling features (pts_rolling_3/5/10 -- cf.
    build_rolling_features). Purement indicatif pour l'affichage : n'entre
    dans aucun vecteur de features et n'affecte pas les predictions xP.
    """
    if gw <= 2:
        return 30   # rolling features quasi vides
    elif gw <= 5:
        return 55   # rolling_3 partiel
    elif gw <= 8:
        return 80   # rolling_5 complet
    else:
        return 95   # rolling_10 complet, confiance max


# 25 features consommees par model_v6_*.pkl (ordre = shap_v6_top_features.pkl)
MODEL_FEATURES_FALLBACK = [
    'minutes', 'value', 'ict_index', 'transfers_in', 'influence',
    'pts_rolling_10', 'pts_rolling_5', 'selected', 'bps', 'transfers_out',
    'pts_rolling_1', 'mins_pct', 'threat', 'bps_rolling_3', 'expected_minutes',
    'form_momentum', 'was_home', 'team_avg_pts_scored', 'saves',
    'pts_rolling_3', 'value_score', 'gw_normalized', 'xa_rolling_3',
    'consistency', 'creativity',
]

# Normalisation des noms d'equipe (copie de best_model_v6.py, pour la
# lecture de cache_elo.pkl qui est indexe par cle canonique)
_NM = {
    "man utd": "man utd", "man united": "man utd", "manchester united": "man utd",
    "man city": "man city", "manchester city": "man city",
    "nott'm forest": "nott'm forest", "nottm forest": "nott'm forest",
    "nottingham forest": "nott'm forest",
    "sheffield utd": "sheffield united", "sheffield united": "sheffield united",
    "aston villa": "aston villa",
    "west ham": "west ham", "west ham united": "west ham",
    "wolves": "wolves", "wolverhampton wanderers": "wolves",
    "brighton": "brighton", "brighton & hove albion": "brighton",
    "crystal palace": "crystal palace",
    "leeds": "leeds", "leeds united": "leeds",
    "leicester": "leicester", "leicester city": "leicester",
    "newcastle": "newcastle", "newcastle united": "newcastle",
    "norwich": "norwich", "norwich city": "norwich",
    "luton": "luton", "luton town": "luton",
    "ipswich": "ipswich", "ipswich town": "ipswich",
    "west brom": "west brom", "west bromwich albion": "west brom",
    "huddersfield": "huddersfield",
    "cardiff": "cardiff", "cardiff city": "cardiff",
    "swansea": "swansea", "swansea city": "swansea",
    "manunited": "man utd", "mancity": "man city",
    "astonvilla": "aston villa", "westham": "west ham",
    "nottmforest": "nott'm forest", "crystalpalace": "crystal palace",
    "sheffieldunited": "sheffield united", "westbrom": "west brom",
}


def norm_team(t) -> str:
    s = str(t).strip().lower()
    return _NM.get(s, s)


def latest_elo(team_name: str, cache_elo: dict, default: float = 1500.0) -> float:
    """Derniere cote Elo connue pour une equipe (cache_elo.pkl est une serie
    temporelle continue par equipe, pas bornee par saison -- couvre donc
    deja 2025-26 pour les equipes deja vues ; retombe sur `default` pour une
    equipe absente (ex: equipe promue jamais rencontree, Coventry/Hull)."""
    if not isinstance(cache_elo, dict):
        return default
    key = norm_team(team_name)
    entry = cache_elo.get(key)
    if not entry:
        return default
    try:
        _dates, elos = entry
        return float(elos[-1]) if len(elos) > 0 else default
    except Exception:
        return default


def detect_promoted_teams(df_train: pd.DataFrame, teams_current: set) -> set:
    """Equipes presentes dans l'API actuelle mais absentes de la DERNIERE
    saison du cache (pas de tout l'historique -- sinon une equipe promue
    l'an dernier et toujours en PL serait signalee a tort)."""
    last_season = sorted(df_train['season'].unique())[-1]
    teams_last_season = set(df_train[df_train['season'] == last_season]['team'].unique())
    return set(teams_current) - teams_last_season


# Equipes promues en Premier League par saison (pour is_promoted_team et
# le fallback "equipe promue" -- verifie/complete au fil des saisons)
PROMOTED_TEAMS_BY_SEASON = {
    '2016-17': ['Burnley', 'Middlesbrough', 'Hull'],
    '2017-18': ['Newcastle', 'Brighton', 'Huddersfield'],
    '2018-19': ['Wolves', 'Cardiff', 'Fulham'],
    '2019-20': ['Norwich', 'Sheffield Utd', 'Aston Villa'],
    '2020-21': ['Leeds', 'West Brom', 'Fulham'],
    '2021-22': ['Norwich', 'Watford', 'Brentford'],
    '2022-23': ["Nott'm Forest", 'Fulham', 'Bournemouth'],
    '2023-24': ['Burnley', 'Sheffield Utd', 'Luton'],
    '2024-25': ['Ipswich', 'Leicester', 'Southampton'],
    '2025-26': ['Leeds', 'Burnley', 'Sunderland'],
    '2026-27': ['Coventry', 'Ipswich', 'Hull'],
}


# =============================================================================
# Section 1 -- Telechargement / standardisation (copie de best_model_v5.py)
# =============================================================================
def fetch_player_positions(season: str) -> pd.DataFrame:
    """Download players_raw.csv and return df with element, position, team."""
    url = PLAYERS_URL.format(season=season)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df_p = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        id_col = 'id' if 'id' in df_p.columns else 'code'
        df_p['_elem_id'] = pd.to_numeric(df_p[id_col], errors='coerce')
        if 'element_type' in df_p.columns:
            df_p['_pos'] = df_p['element_type'].map(
                lambda x: POS_MAP.get(str(x).strip(), None))
        elif 'position' in df_p.columns:
            df_p['_pos'] = df_p['position'].map(
                lambda x: POS_MAP.get(str(x).strip(), None))
        else:
            df_p['_pos'] = None
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
    if 'position' not in df_s.columns or df_s['position'].isna().all():
        if len(player_info) > 0 and 'element' in df_s.columns:
            elem_to_pos = dict(zip(player_info['_elem_id'], player_info['_pos']))
            elem_to_team = dict(zip(player_info['_elem_id'], player_info['_team']))
            df_s['element'] = pd.to_numeric(df_s['element'], errors='coerce')
            df_s['position'] = df_s['element'].map(elem_to_pos)
            if 'team' not in df_s.columns:
                df_s['team'] = df_s['element'].map(elem_to_team).fillna(f'unk_{season}')
    else:
        df_s['position'] = df_s['position'].map(
            lambda x: POS_MAP.get(str(x).strip(), None))

    if 'value' not in df_s.columns:
        df_s['value'] = df_s['now_cost'] if 'now_cost' in df_s.columns else 0.0

    if 'was_home' in df_s.columns:
        df_s['was_home'] = df_s['was_home'].map(
            {True: 1, False: 0, 'True': 1, 'False': 0,
             1: 1, 0: 0, '1': 1, '0': 0}
        ).fillna(0).astype(int)
    else:
        df_s['was_home'] = 0

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

    if 'team' not in df_s.columns:
        df_s['team'] = f'unk_{season}'
    else:
        df_s['team'] = df_s['team'].astype(str).str.strip()

    df_s['opponent_team'] = (
        pd.to_numeric(df_s['opponent_team'], errors='coerce')
        .fillna(0).astype(int).astype(str)
    )

    df_s['season'] = season
    return df_s


def download_season(season: str) -> pd.DataFrame:
    """Download + standardise one season of merged_gw.csv. Returns POSITIONS-filtered df."""
    resp = requests.get(BASE_URL.format(season=season), timeout=30)
    resp.raise_for_status()
    df_s = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    needs_pos = 'position' not in df_s.columns or df_s['position'].isna().all()
    player_info = fetch_player_positions(season) if needs_pos else pd.DataFrame()
    df_s = standardise(df_s, season, player_info)
    df_s = df_s[df_s['position'].isin(POSITIONS)].copy()
    return df_s


# =============================================================================
# Section 2 -- Feature engineering (copie de best_model_v5.py section 2)
# =============================================================================
def build_rolling_features(df_all: pd.DataFrame, cache_merged_v2_path='cache_merged_v2.pkl'):
    """
    Reproduit exactement best_model_v5.py section 2.
    Retourne (df_full, df_train) :
      - df_full  : toutes les lignes, avec toutes les features, SANS drop du
                   target manquant (target=NaN sur la derniere GW de chaque
                   saison). C'est la table a utiliser pour lire le
                   "dernier etat connu" d'un joueur (inference).
      - df_train : df_full avec les lignes target=NaN supprimees (= ce que
                   best_model_v5/v6 utilisent pour l'entrainement).
    """
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

    df['pts_rolling_1'] = roll('total_points', 1)
    df['pts_rolling_3'] = roll('total_points', 3)
    df['pts_rolling_5'] = roll('total_points', 5)
    df['pts_rolling_10'] = roll('total_points', 10)
    df['mins_rolling_3'] = roll('minutes', 3)
    df['bps_rolling_3'] = roll('bps', 3)
    df['goals_rolling_5'] = roll('goals_scored', 5)
    df['assists_rolling_5'] = roll('assists', 5)
    df['xg_rolling_3'] = roll('expected_goals', 3)
    df['xg_rolling_5'] = roll('expected_goals', 5)
    df['xa_rolling_3'] = roll('expected_assists', 3)
    df['xgc_rolling_5'] = roll('expected_goals_conceded', 5)

    df['form_momentum'] = df['pts_rolling_3'] - df['pts_rolling_10']
    df['value_score'] = (
        df['pts_rolling_5'] / (df['value'].clip(lower=0.1) + 0.1)
    ).clip(0, 10)
    pts_std5 = roll('total_points', 5, func='std')
    df['consistency'] = (1 - pts_std5 / (df['pts_rolling_5'] + 0.1)).clip(-2, 1)
    df['big_game_flag'] = np.where(df['was_home'] == 1, 1.15, 0.85)
    df['net_transfers'] = df['transfers_in'] - df['transfers_out']
    df['mins_pct'] = (df['mins_rolling_3'] / 90.0).clip(0, 1)

    df['is_endseason'] = (df['GW'] >= 33).astype(int)
    df['is_earlyseason'] = (df['GW'] <= 6).astype(int)
    df['gw_normalized'] = df['GW'] / 38.0
    if 'season_order' not in df.columns:
        df['season_order'] = df['season'].map(SEASON_ORDER).fillna(0).astype(int)

    df_s2 = df.sort_values(['team', 'season', 'GW']).copy()
    tgw = df_s2.groupby(['season', 'team', 'GW']).agg(
        _team_pts=('total_points', 'mean'),
        _team_goals=('goals_scored', 'sum'),
        _team_cs=('clean_sheets', 'max'),
    ).reset_index()
    for src, dst in [('_team_pts', 'team_avg_pts_scored'),
                      ('_team_goals', 'team_goals_per_game'),
                      ('_team_cs', 'team_clean_sheet_rate')]:
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

    if os.path.exists(cache_merged_v2_path):
        df_v2 = pickle.load(open(cache_merged_v2_path, 'rb'))
        q_src = df_v2[['name', 'season', 'GW'] + QUALITY_COLS].copy()
        for c in QUALITY_COLS:
            q_src[c] = pd.to_numeric(q_src[c], errors='coerce').fillna(0)
        # cache_merged_v2.pkl contient des doublons (name, season, GW) pour
        # certaines GW (ex: 374 doublons en 2024-25, concentres GW24/25/32/33
        # -- source Understat). Sans dedupe, le merge many-to-one devient
        # many-to-many et DOUBLE les lignes de df pour ces GW precises,
        # ce qui gonfle artificiellement leur poids a l'entrainement et
        # degrade le top10_accuracy a l'evaluation (des GW entieres se
        # retrouvent avec ~30-60% de lignes dupliquees).
        q_src = q_src.drop_duplicates(subset=['name', 'season', 'GW'], keep='first')
        df = df.merge(q_src, on=['name', 'season', 'GW'], how='left')
    for c in QUALITY_COLS:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(0)

    df = df.sort_values(['name', 'season', 'GW'])
    df['target'] = df.groupby(['name', 'season'])['total_points'].transform(
        lambda x: x.shift(-1)
    )
    df_full = df.reset_index(drop=True)
    df_train = df_full.dropna(subset=['target']).copy()
    df_train['target'] = df_train['target'].astype(float)
    return df_full, df_train


# =============================================================================
# A1 -- Disponibilite / minutes attendues (copie de best_model_v6.py)
# =============================================================================
def build_availability_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """Reproduit best_model_v6.py section A1 (sans l'override live 2024-25,
    qui ne s'applique qu'aux lignes deja jouees -- inutile pour un backfill
    de saison terminee)."""
    avail_df = df_full[['name', 'season', 'GW', 'minutes', 'mins_rolling_3',
                         'team']].copy()
    avail_df['games_this_gw'] = (1 + (avail_df['minutes'] > 90).astype(int)).astype(float)
    avail_df['is_double_gw'] = (avail_df['minutes'] > 90).astype(int)
    avail_df['is_blank_gw'] = 0
    avail_df['chance_of_playing'] = (avail_df['mins_rolling_3'] / 90.0).clip(0, 1)
    avail_df['expected_minutes'] = (
        avail_df['mins_rolling_3']
        * avail_df['chance_of_playing']
        * avail_df['games_this_gw'].clip(1, 2)
    )
    return avail_df


AVAIL_COLS = ['name', 'season', 'GW', 'games_this_gw', 'is_double_gw',
              'is_blank_gw', 'chance_of_playing', 'expected_minutes']

EXTRA_V6_COLS = [
    ('prob_team_win', 0.0), ('prob_draw', 0.0), ('prob_over25', 0.0),
    ('prob_clean_sheet_approx', 0.0), ('odds_goals_context', 0.0),
    ('elo_team', 1500.0), ('elo_opponent', 1500.0), ('elo_diff', 0.0),
    ('elo_win_prob', 0.5),
]


def carry_forward_extra_cols(df_new: pd.DataFrame, df_old: pd.DataFrame,
                              extra_cols: list = None) -> pd.DataFrame:
    """Reporte les colonnes odds/elo (non recalculees a chaque run, ni
    utilisees par le modele -- cf. shap_v6_top_features.pkl) de l'ancien
    cache_v6_features.pkl vers le nouveau, pour ne pas les perdre a chaque
    backfill/retrain. Les lignes sans correspondance (nouvelle GW) retombent
    sur le meme fallback neutre que best_model_v6.py."""
    extra_cols = extra_cols or EXTRA_V6_COLS
    if df_old is not None and len(df_old) > 0:
        lut_cols = [c for c, _ in extra_cols if c in df_old.columns]
        if lut_cols:
            lut = (df_old[['name', 'season', 'GW'] + lut_cols]
                   .drop_duplicates(subset=['name', 'season', 'GW'], keep='first'))
            df_new = df_new.merge(lut, on=['name', 'season', 'GW'], how='left')
    for c, default in extra_cols:
        if c not in df_new.columns:
            df_new[c] = default
        df_new[c] = pd.to_numeric(df_new[c], errors='coerce').fillna(default)
    return df_new


def merge_availability(df_full: pd.DataFrame, avail_df: pd.DataFrame) -> pd.DataFrame:
    df = df_full.merge(avail_df[AVAIL_COLS], on=['name', 'season', 'GW'], how='left')
    for c in ['games_this_gw', 'is_double_gw', 'is_blank_gw',
              'chance_of_playing', 'expected_minutes']:
        df[c] = df[c].fillna(0 if c != 'chance_of_playing' else 1.0)
    return df


# =============================================================================
# Latest known state per player (pour inference / fallback)
# =============================================================================
def latest_player_snapshot(df_full_avail: pd.DataFrame) -> pd.DataFrame:
    """Un row par joueur = sa derniere ligne (name+season+GW la plus recente,
    en s'appuyant sur season_order puis GW). df_full_avail doit contenir
    season_order (ajoute par build_rolling_features)."""
    d = df_full_avail.sort_values(['name', 'season_order', 'GW'])
    return d.groupby('name', as_index=False).tail(1).reset_index(drop=True)


TEAM_CONTEXT_COLS = ['team_avg_pts_scored', 'team_goals_per_game', 'team_clean_sheet_rate']


def latest_team_snapshot(df_full_avail: pd.DataFrame) -> pd.DataFrame:
    """Un row par equipe = sa forme la plus recente (team_avg_pts_scored,
    team_goals_per_game, team_clean_sheet_rate). Sert a corriger le contexte
    d'equipe d'un joueur qui a change de club (transfert) : sans ca, un
    joueur transfere garderait le contexte de son ANCIEN club dans son
    dernier snapshot historique."""
    d = df_full_avail.sort_values(['team', 'season_order', 'GW'])
    snap = d.groupby('team', as_index=False).tail(1).reset_index(drop=True)
    return snap[['team'] + TEAM_CONTEXT_COLS]


# =============================================================================
# Fallback nouveaux joueurs / equipes promues
# =============================================================================
def position_price_bucket_average(df_train: pd.DataFrame, position: str,
                                   price_m: float, feature_cols: list,
                                   band: float = 0.5) -> dict:
    """Moyenne (ponderee par proximite de prix) des features pour les joueurs
    de meme position dans une tranche de prix [price_m-band, price_m+band].
    `price_m` en millions (ex: 5.5), `df_train['value']` est en dixiemes de
    million (ex: 55) -> on convertit pour comparer."""
    d = df_train[df_train['position'] == position].copy()
    d['_price_m'] = d['value'] / 10.0
    lo, hi = price_m - band, price_m + band
    d = d[(d['_price_m'] >= lo) & (d['_price_m'] <= hi)]
    if len(d) == 0:
        # elargit progressivement si aucune ligne dans la bande
        for extra in (1.0, 2.0, 5.0):
            d = df_train[df_train['position'] == position].copy()
            d['_price_m'] = d['value'] / 10.0
            d = d[(d['_price_m'] >= price_m - band - extra) &
                  (d['_price_m'] <= price_m + band + extra)]
            if len(d) > 0:
                break
    if len(d) == 0:
        return {c: 0.0 for c in feature_cols}
    # ponderation : plus proche du prix cible => poids plus fort
    w = 1.0 / (1.0 + (d['_price_m'] - price_m).abs())
    out = {}
    for c in feature_cols:
        if c not in d.columns:
            out[c] = 0.0
            continue
        vals = pd.to_numeric(d[c], errors='coerce').fillna(0)
        out[c] = float(np.average(vals, weights=w))
    return out


def promoted_team_average(df_train: pd.DataFrame, position: str,
                           feature_cols: list,
                           promoted_teams_by_season: dict = None) -> dict:
    """Moyenne des features des joueurs d'equipes promues (saison de
    promotion uniquement, GW<=38) dans les saisons precedentes, pour une
    position donnee -- sert de profil de depart pour une equipe qui vient
    de monter."""
    promoted_teams_by_season = promoted_teams_by_season or PROMOTED_TEAMS_BY_SEASON
    rows = []
    for season, teams in promoted_teams_by_season.items():
        if season not in set(df_train['season'].unique()):
            continue
        sub = df_train[(df_train['season'] == season) &
                        (df_train['team'].isin(teams)) &
                        (df_train['position'] == position)]
        rows.append(sub)
    if not rows or sum(len(r) for r in rows) == 0:
        return {c: 0.0 for c in feature_cols}
    d = pd.concat(rows, ignore_index=True)
    out = {}
    for c in feature_cols:
        if c not in d.columns:
            out[c] = 0.0
            continue
        out[c] = float(pd.to_numeric(d[c], errors='coerce').fillna(0).mean())
    return out


def predict_new_player(position: str, price_m: float, df_train: pd.DataFrame,
                        feature_cols: list, is_promoted_team: bool = False,
                        price_band: float = 0.5) -> dict:
    """Fallback feature vector pour un joueur SANS historique dans nos
    saisons (nouveau transfert, jeune, ou joueur d'une equipe qui vient
    d'etre promue).

    - Equipe promue (is_promoted_team=True) : moyenne des joueurs des
      equipes promues precedentes (meme position) dans nos 10 saisons --
      typiquement des performances plus faibles que la moyenne de la ligue,
      ce qui reflete correctement la 1ere saison d'un promu.
    - Sinon : moyenne des joueurs de meme position ET meme tranche de prix
      (+/- price_band millions), ponderee par proximite de prix -- le prix
      de depart fixe par FPL reflete deja l'avis des recruteurs/analystes
      FPL sur le niveau attendu du joueur.

    Retourne un dict {feature_name: valeur} pret a etre assemble en vecteur
    de prediction, plus la cle 'is_promoted_team' (0/1).
    """
    if is_promoted_team:
        feat = promoted_team_average(df_train, position, feature_cols)
    else:
        feat = position_price_bucket_average(df_train, position, price_m,
                                              feature_cols, band=price_band)
    feat['value'] = price_m * 10.0  # 'value' est en dixiemes de million dans le cache
    feat['is_promoted_team'] = 1 if is_promoted_team else 0
    return feat


# =============================================================================
# Construction du vecteur de features "live" pour un joueur (inference)
# =============================================================================
def build_live_feature_vector(player_key: str, position: str, price_m: float,
                               team_name: str, was_home: bool,
                               chance_of_playing_next: float, games_next_gw: int,
                               gw_number: int, feature_cols: list,
                               df_latest: pd.DataFrame, df_train: pd.DataFrame,
                               team_snapshot: pd.DataFrame,
                               is_promoted_team: bool) -> tuple:
    """Construit le vecteur de features pour predire la GW `gw_number` d'un
    joueur, en reutilisant son dernier etat connu (df_latest) quand il
    existe, ou le fallback predict_new_player() sinon.

    Retourne (feat_dict, has_history: bool).
    """
    row = df_latest[df_latest['name'] == player_key]

    if len(row) > 0:
        has_history = True
        r = row.iloc[0]
        feat = {c: float(r[c]) if c in row.columns and pd.notna(r[c]) else 0.0
                for c in feature_cols}
        # 'value' (prix) : utiliser le prix ACTUEL, pas le dernier prix connu
        feat['value'] = price_m * 10.0
        # mins_rolling_3 du joueur sert de base pour expected_minutes live
        mins_rolling_3 = float(r.get('mins_rolling_3', 0.0))
    else:
        has_history = False
        feat = predict_new_player(position, price_m, df_train, feature_cols,
                                   is_promoted_team=is_promoted_team)
        mins_rolling_3 = feat.get('mins_rolling_3', 0.0)

    # -- Contexte d'equipe : toujours celui de l'equipe ACTUELLE (gere les
    #    transferts et les equipes promues), jamais celui du dernier
    #    snapshot historique (qui peut etre une ancienne equipe) ---------
    trow = team_snapshot[team_snapshot['team'] == team_name]
    if len(trow) > 0:
        for c in TEAM_CONTEXT_COLS:
            if c in feature_cols:
                feat[c] = float(trow.iloc[0][c])
    elif is_promoted_team:
        promoted_ctx = promoted_team_average(df_train, position, TEAM_CONTEXT_COLS)
        for c in TEAM_CONTEXT_COLS:
            if c in feature_cols:
                feat[c] = promoted_ctx.get(c, 0.0)

    # -- Contexte du prochain match (toujours live, jamais historique) ---
    if 'was_home' in feature_cols:
        feat['was_home'] = 1.0 if was_home else 0.0
    if 'gw_normalized' in feature_cols:
        feat['gw_normalized'] = gw_number / 38.0
    if 'expected_minutes' in feature_cols:
        feat['expected_minutes'] = (
            mins_rolling_3 * float(chance_of_playing_next) *
            min(max(games_next_gw, 1), 2)
        )

    return feat, has_history
