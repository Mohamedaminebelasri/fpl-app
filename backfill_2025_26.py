"""
backfill_2025_26.py

Le cache cache_9seasons_features.pkl s'arretait a la saison 2024-25 (GW37).
Or nous sommes en aout 2026 : la saison 2025-26 complete (38 GW) est absente
du cache alors qu'elle vient de se terminer. Consequence concrete : pour
QUASIMENT TOUS les joueurs revenants en 2026-27 (pas seulement les
transferts/promus), les rolling features (pts_rolling_5, form_momentum,
etc.) seraient calculees sur des donnees vieilles de plus d'un an.

Ce script :
  1. Telecharge la saison 2025-26 (vaastav/Fantasy-Premier-League, meme
     source que best_model_v5.py) et l'ajoute a cache_9seasons.pkl.
  2. Recalcule les features (rolling / interaction / team / opponent /
     quality / target) sur les 10 saisons -> cache_9seasons_features.pkl.
  3. Recalcule les features de disponibilite (games_this_gw, is_double_gw,
     chance_of_playing, expected_minutes) -> cache_availability.pkl.
  4. Refusionne dans cache_v6_features.pkl (reutilise cache_odds.pkl /
     cache_elo.pkl existants tels quels : elo couvre deja 2025-26 pour les
     equipes connues car recupere par serie temporelle continue depuis
     clubelo.com, pas par saison ; les odds 2025-26 ne seront pas
     matchees et retombent sur le fallback 0 deja utilise pour les lignes
     sans donnee -- ni l'un ni l'autre n'est dans le top-25 de features du
     modele, donc aucun impact sur les predictions xP).
  5. Sauvegarde cache_player_latest_state.pkl : une ligne par joueur =
     son dernier match connu, toutes features calculees -- c'est cette
     table que get_xp_predictions() utilise pour "l'etat actuel" d'un
     joueur avant de predire sa GW1 2026-27.

Idempotent : peut etre relance sans dupliquer la saison si elle est deja
presente. Sauvegarde les caches existants en .bak avant ecrasement.
"""
import os
import pickle
import shutil
import time

import pandas as pd
import fpl_feature_lib as flib

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

NEW_SEASON = '2025-26'

CACHE_9S = 'cache_9seasons.pkl'
CACHE_FEAT = 'cache_9seasons_features.pkl'
CACHE_AVAIL = 'cache_availability.pkl'
CACHE_V6 = 'cache_v6_features.pkl'
CACHE_LATEST = 'cache_player_latest_state.pkl'


def backup(path):
    if os.path.exists(path) and not os.path.exists(path + '.bak_pre_2025_26'):
        print(f"   Backup {path} -> {path}.bak_pre_2025_26")
        shutil.copy2(path, path + '.bak_pre_2025_26')


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"   BACKFILL SAISON {NEW_SEASON} -> caches V6 (prep 2026/27)")
    print("=" * 70)

    # ---- 1. Raw season data ------------------------------------------------
    print(f"\n[1/5] cache_9seasons.pkl ...")
    df_all = pickle.load(open(CACHE_9S, 'rb'))
    seasons_present = set(df_all['season'].unique())

    if NEW_SEASON in seasons_present:
        print(f"   {NEW_SEASON} deja presente ({len(df_all[df_all['season']==NEW_SEASON]):,} lignes) -- skip telechargement.")
    else:
        print(f"   Telechargement {NEW_SEASON} depuis vaastav/Fantasy-Premier-League ...")
        df_new = flib.download_season(NEW_SEASON)
        print(f"   {len(df_new):,} lignes telechargees.")
        backup(CACHE_9S)
        df_all = pd.concat([df_all, df_new], ignore_index=True, sort=False)
        df_all['season_order'] = df_all['season'].map(flib.SEASON_ORDER).fillna(0).astype(int)
        pickle.dump(df_all, open(CACHE_9S, 'wb'))
        print(f"   Sauvegarde {CACHE_9S}: {len(df_all):,} lignes, saisons={sorted(df_all['season'].unique())}")

    # ---- 2. Feature engineering --------------------------------------------
    print(f"\n[2/5] cache_9seasons_features.pkl (rolling/interaction/team/target) ...")
    df_feat_existing = pickle.load(open(CACHE_FEAT, 'rb')) if os.path.exists(CACHE_FEAT) else None
    need_rebuild = (df_feat_existing is None) or (NEW_SEASON not in set(df_feat_existing['season'].unique()))

    if not need_rebuild:
        print(f"   {NEW_SEASON} deja dans {CACHE_FEAT} -- skip recalcul.")
        df_full, df_train = None, df_feat_existing
    else:
        print("   Recalcul complet sur 10 saisons (rolling features, ~1-2 min) ...")
        df_full, df_train = flib.build_rolling_features(df_all)
        backup(CACHE_FEAT)
        pickle.dump(df_train, open(CACHE_FEAT, 'wb'))
        print(f"   Sauvegarde {CACHE_FEAT}: {len(df_train):,} lignes (target non-null), {df_train.shape[1]} colonnes")

    # ---- 3. Availability features -------------------------------------------
    print(f"\n[3/5] cache_availability.pkl ...")
    if df_full is None:
        # need the undropped version to recompute availability + latest snapshot
        df_full, _ = flib.build_rolling_features(df_all)

    avail_existing = pickle.load(open(CACHE_AVAIL, 'rb')) if os.path.exists(CACHE_AVAIL) else None
    avail_need_rebuild = (avail_existing is None) or (NEW_SEASON not in set(avail_existing['season'].unique()))
    if not avail_need_rebuild:
        print(f"   {NEW_SEASON} deja dans {CACHE_AVAIL} -- skip recalcul.")
        avail_df = avail_existing
    else:
        avail_df = flib.build_availability_features(df_full)
        backup(CACHE_AVAIL)
        pickle.dump(avail_df, open(CACHE_AVAIL, 'wb'))
        print(f"   Sauvegarde {CACHE_AVAIL}: {len(avail_df):,} lignes")

    # ---- 4. cache_v6_features.pkl (fusion) ----------------------------------
    print(f"\n[4/5] cache_v6_features.pkl (fusion availability + odds/elo existants) ...")
    df_v6_existing = pickle.load(open(CACHE_V6, 'rb')) if os.path.exists(CACHE_V6) else None
    v6_need_rebuild = (df_v6_existing is None) or (NEW_SEASON not in set(df_v6_existing['season'].unique()))
    if not v6_need_rebuild:
        print(f"   {NEW_SEASON} deja dans {CACHE_V6} -- skip fusion.")
        df_v6 = df_v6_existing
    else:
        df_v6 = flib.merge_availability(df_full, avail_df)
        # Reutilise les colonnes odds/elo deja presentes dans l'ancien cache_v6
        # pour les lignes deja existantes ; nouvelles lignes 2025-26 -> fallback
        # neutre (0 pour odds, 1500/0 pour elo), comme le fait deja
        # best_model_v6.py pour toute ligne sans correspondance.
        extra_cols = [('prob_team_win', 0.0), ('prob_draw', 0.0),
                      ('prob_over25', 0.0), ('prob_clean_sheet_approx', 0.0),
                      ('odds_goals_context', 0.0),
                      ('elo_team', 1500.0), ('elo_opponent', 1500.0),
                      ('elo_diff', 0.0), ('elo_win_prob', 0.5)]
        if df_v6_existing is not None:
            lut_cols = [c for c, _ in extra_cols if c in df_v6_existing.columns]
            if lut_cols:
                lut = (df_v6_existing[['name', 'season', 'GW'] + lut_cols]
                       .drop_duplicates(subset=['name', 'season', 'GW'], keep='first'))
                df_v6 = df_v6.merge(lut, on=['name', 'season', 'GW'], how='left')
        for c, default in extra_cols:
            if c not in df_v6.columns:
                df_v6[c] = default
            df_v6[c] = pd.to_numeric(df_v6[c], errors='coerce').fillna(default)
        backup(CACHE_V6)
        pickle.dump(df_v6, open(CACHE_V6, 'wb'))
        print(f"   Sauvegarde {CACHE_V6}: {len(df_v6):,} lignes, {df_v6.shape[1]} colonnes")

    # ---- 5. Latest known state per player -----------------------------------
    print(f"\n[5/5] {CACHE_LATEST} (dernier etat connu par joueur, pour inference) ...")
    df_latest = flib.latest_player_snapshot(df_v6)
    pickle.dump(df_latest, open(CACHE_LATEST, 'wb'))
    print(f"   Sauvegarde {CACHE_LATEST}: {len(df_latest):,} joueurs (1 ligne/joueur)")
    print(f"   Dernieres saisons representees: "
          f"{df_latest.groupby('season')['name'].count().sort_index().tail(3).to_dict()}")

    print(f"\nOK -- backfill termine en {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
