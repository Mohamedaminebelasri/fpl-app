"""
retrain_weekly.py -- Ingestion hebdomadaire + reentrainement rapide V6
Saison 2026/27

A chaque execution :
  1. Verifie la derniere GW terminee via l'API FPL (bootstrap-static).
  2. Pour chaque GW terminee pas encore integree :
       a. Telecharge ses stats (event/{gw}/live/) + fixtures.
       b. Les ajoute a cache_9seasons.pkl, recalcule les features
          (rolling, disponibilite, fusion) -> cache_9seasons_features.pkl /
          cache_availability.pkl / cache_v6_features.pkl /
          cache_player_latest_state.pkl.
       c. Evalue les xP predits la semaine derniere pour cette GW (dans
          predictions_log.csv) contre les points reels obtenus -> logge
          MAE/RMSE dans accuracy_log.csv (dashboard precision, Tache 4).
       d. Reentraine les 4 modeles XGBoost avec les hyperparametres Optuna
          existants (optuna_XGB_*.pkl), fit() simple -- pas de recherche
          d'hyperparametres, <5 min.
       e. Sauvegarde model_v6_{POS}_gw{N}.pkl (versionne) + model_v6_{POS}.pkl
          (copie "latest" utilisee par l'app).
       f. Genere les xP pour la GW suivante avec le modele frais, les
          sauvegarde dans predictions_log.csv (pour l'evaluation de la
          semaine prochaine).
  3. Logge chaque etape dans retrain_log.txt.

Idempotent : relancer plusieurs fois sans nouvelle GW terminee ne fait rien.
Usage : python retrain_weekly.py   (a planifier ex. cron/Task Scheduler,
apres chaque GW, ou en loop hebdomadaire).
"""
import os
import io
import json
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import fpl_feature_lib as flib

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

SEASON_CURRENT = '2026-27'
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
FIXED_PARAMS = dict(n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')
HTTP_HEADERS = {'User-Agent': 'Mozilla/5.0 (FPL-Retrain-Weekly)'}

STATE_FILE = 'retrain_state.json'
PRED_LOG = 'predictions_log.csv'
ACC_LOG = 'accuracy_log.csv'
RETRAIN_LOG = 'retrain_log.txt'

CACHE_9S = 'cache_9seasons.pkl'
CACHE_FEAT = 'cache_9seasons_features.pkl'
CACHE_AVAIL = 'cache_availability.pkl'
CACHE_V6 = 'cache_v6_features.pkl'
CACHE_LATEST = 'cache_player_latest_state.pkl'
SHAP_FEATURES = 'shap_v6_top_features.pkl'

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVE_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"


# =============================================================================
# Logging / state
# =============================================================================
def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(RETRAIN_LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'last_gw_ingested': 0}


def save_state(state: dict):
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# =============================================================================
# API FPL
# =============================================================================
def fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=20, headers=HTTP_HEADERS)
    resp.raise_for_status()
    return resp.json()


def build_gw_rows(gw: int, bootstrap: dict, fixtures_gw: list, live: dict) -> pd.DataFrame:
    """Assemble des lignes au format merged_gw (une ligne/joueur) pour une
    GW terminee, a partir de event/{gw}/live/ + fixtures + bootstrap-static."""
    teams_dict = {t['id']: t['name'] for t in bootstrap['teams']}
    elem_lookup = {e['id']: e for e in bootstrap['elements']}
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    team_fixture = {}
    for f in fixtures_gw:
        team_fixture[f['team_h']] = (f, True)
        team_fixture[f['team_a']] = (f, False)

    def _num(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    rows = []
    for el in live.get('elements', []):
        pid = el['id']
        p = elem_lookup.get(pid)
        if p is None:
            continue
        stats = el.get('stats', {})
        tid = p['team']
        fx, was_home = team_fixture.get(tid, (None, True))
        opponent_team = (fx['team_a'] if was_home else fx['team_h']) if fx else 0
        kickoff_time = fx.get('kickoff_time') if fx else None

        rows.append({
            'name': f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            'position': pos_map.get(p['element_type'], 'MID'),
            'team': teams_dict.get(tid, ''),
            'element': pid,
            'GW': gw,
            'season': SEASON_CURRENT,
            'total_points': _num(stats.get('total_points')),
            'minutes': _num(stats.get('minutes')),
            'goals_scored': _num(stats.get('goals_scored')),
            'assists': _num(stats.get('assists')),
            'clean_sheets': _num(stats.get('clean_sheets')),
            'goals_conceded': _num(stats.get('goals_conceded')),
            'own_goals': _num(stats.get('own_goals')),
            'penalties_saved': _num(stats.get('penalties_saved')),
            'penalties_missed': _num(stats.get('penalties_missed')),
            'yellow_cards': _num(stats.get('yellow_cards')),
            'red_cards': _num(stats.get('red_cards')),
            'saves': _num(stats.get('saves')),
            'bonus': _num(stats.get('bonus')),
            'bps': _num(stats.get('bps')),
            'influence': _num(stats.get('influence')),
            'creativity': _num(stats.get('creativity')),
            'threat': _num(stats.get('threat')),
            'ict_index': _num(stats.get('ict_index')),
            'expected_goals': _num(stats.get('expected_goals')),
            'expected_assists': _num(stats.get('expected_assists')),
            'expected_goal_involvements': _num(stats.get('expected_goal_involvements')),
            'expected_goals_conceded': _num(stats.get('expected_goals_conceded')),
            'value': p.get('now_cost', 0),
            'selected': _num(p.get('selected_by_percent'), 0.0),
            'transfers_in': _num(p.get('transfers_in_event'), 0.0),
            'transfers_out': _num(p.get('transfers_out_event'), 0.0),
            'was_home': 1 if was_home else 0,
            'opponent_team': opponent_team,
            'kickoff_time': kickoff_time,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Modele
# =============================================================================
def load_optuna(pos: str) -> dict:
    path = f'optuna_XGB_{pos}.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    return dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=2)


def train_and_save_models(df_train: pd.DataFrame, feature_cols: list, gw: int):
    results = {}
    for pos in POSITIONS:
        d = df_train[df_train['position'] == pos]
        if len(d) < 50:
            log(f"   {pos}: seulement {len(d)} lignes -- skip.")
            continue
        X = d[feature_cols].fillna(0).values
        y = d['target'].values
        w = np.where(y >= 8, 1.5, 1.0)
        params = {**load_optuna(pos), **FIXED_PARAMS}
        mdl = XGBRegressor(**params)
        t0 = time.time()
        mdl.fit(X, y, sample_weight=w, verbose=False)
        results[pos] = mdl
        versioned_path = f'model_v6_{pos}_gw{gw}.pkl'
        latest_path = f'model_v6_{pos}.pkl'
        pickle.dump(mdl, open(versioned_path, 'wb'))
        pickle.dump(mdl, open(latest_path, 'wb'))
        log(f"   {pos}: {len(d):,} lignes, fit en {time.time()-t0:.1f}s "
            f"-> {versioned_path} + {latest_path}")
    return results


# =============================================================================
# Dashboard precision (Tache 4) : xP predits vs points reels
# =============================================================================
def evaluate_predictions_for_gw(gw: int, df_new_gw: pd.DataFrame):
    """Compare les xP predits la semaine derniere pour `gw` (predictions_log.csv)
    aux points reels obtenus (df_new_gw, juste ingere) -- log MAE/RMSE."""
    if not os.path.exists(PRED_LOG):
        log(f"   Pas de {PRED_LOG} -- aucune prediction anterieure a evaluer pour GW{gw}.")
        return

    df_pred = pd.read_csv(PRED_LOG)
    df_pred_gw = df_pred[df_pred['gw'] == gw]
    if len(df_pred_gw) == 0:
        log(f"   Aucune prediction enregistree pour GW{gw} -- rien a evaluer.")
        return

    actual = df_new_gw[['element', 'total_points']].rename(
        columns={'element': 'id', 'total_points': 'actual_points'})
    merged = df_pred_gw.merge(actual, on='id', how='inner')
    if len(merged) == 0:
        log(f"   Aucune correspondance id entre predictions GW{gw} et points reels.")
        return

    rows = [{'gw': gw, 'position': 'ALL',
             'mae': mean_absolute_error(merged['actual_points'], merged['xP_pred']),
             'rmse': mean_squared_error(merged['actual_points'], merged['xP_pred']) ** 0.5,
             'n_players': len(merged),
             'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]
    for pos in POSITIONS:
        sub = merged[merged['position'] == pos]
        if len(sub) == 0:
            continue
        rows.append({'gw': gw, 'position': pos,
                      'mae': mean_absolute_error(sub['actual_points'], sub['xP_pred']),
                      'rmse': mean_squared_error(sub['actual_points'], sub['xP_pred']) ** 0.5,
                      'n_players': len(sub),
                      'evaluated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

    df_acc = pd.DataFrame(rows)
    header = not os.path.exists(ACC_LOG)
    df_acc.to_csv(ACC_LOG, mode='a', header=header, index=False)
    overall = rows[0]
    log(f"   Precision GW{gw}: MAE={overall['mae']:.2f}  RMSE={overall['rmse']:.2f}  "
        f"(n={overall['n_players']} joueurs) -> {ACC_LOG}")


# =============================================================================
# Generation des predictions pour la GW suivante (pour evaluation future)
# =============================================================================
def generate_predictions_for_next_gw(next_gw: int, bootstrap: dict, fixtures: list,
                                      models: dict, feature_cols: list,
                                      df_train: pd.DataFrame, df_latest: pd.DataFrame):
    teams_dict = {t['id']: t['name'] for t in bootstrap['teams']}
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    teams_current = set(teams_dict.values())
    promoted_teams = flib.detect_promoted_teams(df_train, teams_current)
    team_snapshot = flib.latest_team_snapshot(df_train)

    upcoming = [f for f in fixtures if f.get('event') == next_gw]
    team_next_fixes, team_next_count = {}, {}
    for f in upcoming:
        for side in ['team_h', 'team_a']:
            tid = f[side]
            team_next_fixes.setdefault(tid, f)
            team_next_count[tid] = team_next_count.get(tid, 0) + 1

    rows = []
    for p in bootstrap['elements']:
        tid = p['team']
        pos = pos_map.get(p['element_type'], 'MID')
        team_name = teams_dict.get(tid, '')
        price_m = p.get('now_cost', 0) / 10.0
        player_key = f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()
        is_promoted = team_name in promoted_teams

        fx = team_next_fixes.get(tid)
        was_home = bool(fx and fx['team_h'] == tid)
        games_next_gw = team_next_count.get(tid, 1)

        chance_raw = p.get('chance_of_playing_next_round')
        chance_norm = (float(chance_raw) if chance_raw is not None else 100.0) / 100.0

        feat, _ = flib.build_live_feature_vector(
            player_key=player_key, position=pos, price_m=price_m,
            team_name=team_name, was_home=was_home,
            chance_of_playing_next=chance_norm, games_next_gw=games_next_gw,
            gw_number=next_gw, feature_cols=feature_cols,
            df_latest=df_latest, df_train=df_train,
            team_snapshot=team_snapshot, is_promoted_team=is_promoted,
        )

        mdl = models.get(pos)
        xp_pred = 0.0
        if mdl is not None:
            fvec = np.array([[float(feat.get(fn, 0)) for fn in feature_cols]])
            try:
                xp_pred = max(0.0, float(mdl.predict(fvec)[0]))
            except Exception:
                xp_pred = 0.0

        rows.append({'gw': next_gw, 'id': p['id'], 'name': p['web_name'],
                      'position': pos, 'price': price_m, 'xP_pred': round(xp_pred, 3),
                      'predicted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

    df_out = pd.DataFrame(rows)
    header = not os.path.exists(PRED_LOG)
    # evite les doublons si le script est relance pour la meme GW
    if os.path.exists(PRED_LOG):
        df_existing = pd.read_csv(PRED_LOG)
        df_existing = df_existing[df_existing['gw'] != next_gw]
        df_out = pd.concat([df_existing, df_out], ignore_index=True)
        df_out.to_csv(PRED_LOG, index=False)
    else:
        df_out.to_csv(PRED_LOG, index=False)
    log(f"   Predictions GW{next_gw} generees pour {len(rows)} joueurs -> {PRED_LOG}")


# =============================================================================
# Ingestion d'une GW terminee
# =============================================================================
def ingest_one_gw(gw: int, bootstrap: dict, fixtures: list, feature_cols: list, state: dict):
    t0 = time.time()
    log(f"--- Ingestion GW{gw} ({SEASON_CURRENT}) ---")

    live = fetch_json(LIVE_URL.format(gw=gw))
    if not live.get('elements'):
        log(f"   GW{gw}: aucune donnee live disponible pour l'instant -- abandon (reessai au prochain run).")
        return False

    fixtures_gw = [f for f in fixtures if f.get('event') == gw]
    df_new = build_gw_rows(gw, bootstrap, fixtures_gw, live)
    df_new = df_new[df_new['position'].isin(POSITIONS)].copy()
    log(f"   {len(df_new):,} lignes joueurs recuperees.")

    # 0. Dashboard precision : evalue les predictions faites la semaine
    #    derniere pour cette GW, AVANT de les ecraser plus bas.
    evaluate_predictions_for_gw(gw, df_new)

    # 1. Cache brut
    df_all = pickle.load(open(CACHE_9S, 'rb'))
    df_all = df_all[~((df_all['season'] == SEASON_CURRENT) & (df_all['GW'] == gw))]
    df_all = pd.concat([df_all, df_new], ignore_index=True, sort=False)
    season_order_ext = dict(flib.SEASON_ORDER)
    if SEASON_CURRENT not in season_order_ext:
        season_order_ext[SEASON_CURRENT] = max(season_order_ext.values()) + 1
    df_all['season_order'] = df_all['season'].map(season_order_ext).fillna(0).astype(int)
    pickle.dump(df_all, open(CACHE_9S, 'wb'))

    # 2. Rolling features (recalcul complet -- ~1-2 min sur ~280k lignes)
    df_full, df_train = flib.build_rolling_features(df_all)
    pickle.dump(df_train, open(CACHE_FEAT, 'wb'))
    log(f"   Rolling features recalculees : {len(df_train):,} lignes train.")

    # 3. Disponibilite
    avail_df = flib.build_availability_features(df_full)
    pickle.dump(avail_df, open(CACHE_AVAIL, 'wb'))

    # 4. Fusion V6 (garde les colonnes odds/elo historiques)
    df_v6_old = pickle.load(open(CACHE_V6, 'rb')) if os.path.exists(CACHE_V6) else None
    df_v6 = flib.merge_availability(df_full, avail_df)
    df_v6 = flib.carry_forward_extra_cols(df_v6, df_v6_old)
    pickle.dump(df_v6, open(CACHE_V6, 'wb'))

    # 5. Dernier etat connu par joueur (pour inference)
    df_latest = flib.latest_player_snapshot(df_v6)
    pickle.dump(df_latest, open(CACHE_LATEST, 'wb'))

    # 6. Reentrainement rapide (fit() simple, hyperparams Optuna fixes).
    #    cache_v6_features.pkl (pas cache_9seasons_features.pkl) car il
    #    contient expected_minutes, une des 25 features du modele.
    df_train_v6 = df_v6.dropna(subset=['target']).copy()
    log(f"   Reentrainement des 4 modeles XGBoost (params Optuna fixes, "
        f"{len(df_train_v6):,} lignes) ...")
    models = train_and_save_models(df_train_v6, feature_cols, gw)

    # 7. Predictions pour la GW suivante (pour evaluation la semaine prochaine)
    next_gw = gw + 1
    if next_gw <= 38:
        generate_predictions_for_next_gw(next_gw, bootstrap, fixtures, models,
                                          feature_cols, df_train, df_latest)

    state['last_gw_ingested'] = gw
    save_state(state)

    log(f"GW{gw} integree — modele mis a jour — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"(duree {time.time()-t0:.1f}s)")
    return True


# =============================================================================
# Main
# =============================================================================
def main():
    log("=" * 60)
    log("retrain_weekly.py -- debut")

    if not os.path.exists(CACHE_FEAT):
        log("cache_9seasons_features.pkl introuvable -- lancez backfill_2025_26.py d'abord. Abandon.")
        return

    top_features = pickle.load(open(SHAP_FEATURES, 'rb')) if os.path.exists(SHAP_FEATURES) else None
    feature_cols = list(top_features) if isinstance(top_features, list) and top_features \
        else list(flib.MODEL_FEATURES_FALLBACK)

    state = load_state()
    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)

    finished_gws = sorted(e['id'] for e in bootstrap['events'] if e.get('finished'))
    if not finished_gws:
        log("Aucune GW terminee pour l'instant (pre-saison) -- rien a faire.")
        return

    latest_finished = max(finished_gws)
    last_ingested = state.get('last_gw_ingested', 0)

    if latest_finished <= last_ingested:
        log(f"GW{latest_finished} deja integree (last_gw_ingested={last_ingested}) -- rien a faire.")
        return

    for gw in range(last_ingested + 1, latest_finished + 1):
        ok = ingest_one_gw(gw, bootstrap, fixtures, feature_cols, state)
        if not ok:
            break  # donnees pas encore dispo, on reessaiera au prochain run

    log("retrain_weekly.py -- fin")


if __name__ == '__main__':
    main()
