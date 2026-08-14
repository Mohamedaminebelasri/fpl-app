"""
diagnostic_2026_27.py

Diagnostic de preparation saison 2026/27 :
  1. Equipes promues (comparees a la VRAIE derniere saison jouee, 2025-26,
     maintenant presente dans le cache grace a backfill_2025_26.py -- une
     comparaison contre l'ensemble des 9-10 saisons historiques donnerait
     de faux positifs, ex: Leeds/Sunderland promus en 2025-26 seraient
     signales a tort comme "nouveaux" alors qu'ils reviennent d'une saison
     qu'on a maintenant en cache).
  2. Joueurs sans historique (nouveaux transferts, jeunes, ou joueurs
     d'equipes promues jamais vues en Premier League dans nos donnees).
  3. Prix de depart FPL 2026/27 pour tous les joueurs.

Usage: python diagnostic_2026_27.py
"""
import pickle

import pandas as pd
import requests

import fpl_feature_lib as flib

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
CACHE_LATEST = 'cache_player_latest_state.pkl'
CACHE_FEAT = 'cache_9seasons_features.pkl'


def fetch_bootstrap():
    resp = requests.get(BOOTSTRAP_URL, timeout=15,
                         headers={'User-Agent': 'Mozilla/5.0 (FPL-Diag-2026-27)'})
    resp.raise_for_status()
    return resp.json()


def player_name_key(p: dict) -> str:
    """Reconstitue le format 'Prenom Nom' utilise par vaastav (colonne
    'name' du cache), a partir de bootstrap-static (first_name/second_name)."""
    return f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()


def main():
    print("=" * 70)
    print("   DIAGNOSTIC SAISON 2026/27")
    print("=" * 70)

    bs = fetch_bootstrap()
    teams_current = sorted(t['name'] for t in bs['teams'])
    teams_dict = {t['id']: t['name'] for t in bs['teams']}

    df_feat = pickle.load(open(CACHE_FEAT, 'rb'))
    df_latest = pickle.load(open(CACHE_LATEST, 'rb')) if __import__('os').path.exists(CACHE_LATEST) else None

    # ---- 1. Equipes promues --------------------------------------------
    last_season = sorted(df_feat['season'].unique())[-1]  # '2025-26'
    teams_last_season = set(df_feat[df_feat['season'] == last_season]['team'].unique())
    promoted = sorted(flib.detect_promoted_teams(df_feat, set(teams_current)))
    relegated = sorted(teams_last_season - set(teams_current))

    print(f"\n[1] Equipes promues (vs derniere saison en cache: {last_season})")
    print(f"    {len(promoted)} equipes promues : {promoted}")
    print(f"    {len(relegated)} equipes releguees (reference) : {relegated}")
    if len(promoted) != 3:
        print(f"    /!\\ Attendu 3 equipes promues, {len(promoted)} trouvees "
              f"-- verifier la coherence cache vs API (saisons manquantes ?).")

    # ---- 2. Joueurs sans historique -------------------------------------
    print(f"\n[2] Joueurs sans historique (jamais vus dans le cache 10 saisons)")
    known_names = set(df_feat['name'].unique())
    no_history, has_history = [], []
    for p in bs['elements']:
        key = player_name_key(p)
        pos = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(p['element_type'], 'MID')
        rec = {
            'id': p['id'], 'name': key, 'team': teams_dict.get(p['team'], '?'),
            'position': pos, 'price_m': p.get('now_cost', 0) / 10.0,
        }
        if key in known_names:
            has_history.append(rec)
        else:
            no_history.append(rec)

    df_no_hist = pd.DataFrame(no_history)
    df_has_hist = pd.DataFrame(has_history)
    print(f"    {len(no_history)} joueurs sans historique / {len(bs['elements'])} au total "
          f"({len(no_history)/len(bs['elements'])*100:.1f}%)")
    if len(df_no_hist) > 0:
        print(f"    Repartition par position :")
        print(df_no_hist['position'].value_counts().to_string().replace('\n', '\n      '))
        print(f"    Repartition par equipe (top 10) :")
        print(df_no_hist['team'].value_counts().head(10).to_string().replace('\n', '\n      '))
        promoted_share = df_no_hist['team'].isin(promoted).mean() * 100
        print(f"    {promoted_share:.0f}% des joueurs sans historique jouent dans une equipe promue.")

    # ---- 3. Prix de depart -----------------------------------------------
    print(f"\n[3] Prix de depart FPL 2026/27")
    all_prices = pd.DataFrame([{
        'id': p['id'],
        'name': player_name_key(p),
        'team': teams_dict.get(p['team'], '?'),
        'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(p['element_type'], 'MID'),
        'price_m': p.get('now_cost', 0) / 10.0,
        'has_history': player_name_key(p) in known_names,
        'is_promoted_team': teams_dict.get(p['team'], '?') in promoted,
    } for p in bs['elements']]).sort_values('price_m', ascending=False)

    print(f"    {len(all_prices)} joueurs, prix de {all_prices['price_m'].min():.1f}m "
          f"a {all_prices['price_m'].max():.1f}m")
    print(f"    Top 5 les plus chers :")
    print(all_prices[['name', 'team', 'position', 'price_m']].head(5)
          .to_string(index=False).replace('\n', '\n      '))

    out_path = 'diagnostic_2026_27_prices.csv'
    all_prices.to_csv(out_path, index=False)
    print(f"\n    Prix complets exportes -> {out_path}")

    # ---- Resume -------------------------------------------------------
    print("\n" + "=" * 70)
    print("   RESUME")
    print("=" * 70)
    print(f"   {len(promoted)} equipes promues : {promoted}")
    print(f"   {len(no_history)} joueurs sans historique / {len(bs['elements'])} "
          f"({len(no_history)/len(bs['elements'])*100:.1f}%)")
    print(f"   Prix de depart exportes pour {len(all_prices)} joueurs -> {out_path}")

    return {
        'promoted_teams': promoted,
        'relegated_teams': relegated,
        'n_no_history': len(no_history),
        'n_total': len(bs['elements']),
        'df_no_history': df_no_hist,
        'df_prices': all_prices,
    }


if __name__ == '__main__':
    main()
