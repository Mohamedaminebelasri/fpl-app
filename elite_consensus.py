"""Elite Consensus — analyse les équipes réelles des top managers mondiaux
(ligue officielle "Overall" id=314) pour compléter les prédictions ML,
particulièrement utile en début de saison (GW1-6) quand le modèle manque
de données historiques.
"""
import json
import math
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fantasy.premierleague.com/api"
OVERALL_LEAGUE_ID = 314
PAGE_SIZE = 50
DATA_DIR = "data"
REQUEST_SLEEP = 0.6  # ~1.6 req/s, respecte la contrainte 1-2 req/s
POS_MAP = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}


def _build_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session


def _cache_path(gameweek):
    return os.path.join(DATA_DIR, f"elite_consensus_gw{gameweek}.json")


def load_cache(gameweek):
    path = _cache_path(gameweek)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_cache(gameweek, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_cache_path(gameweek), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_top_managers(n=1000, session=None, progress_callback=None):
    """Récupère les entry_id des n meilleurs managers de la ligue Overall (314)."""
    session = session or _build_session()
    entries = []
    pages = max(1, math.ceil(n / PAGE_SIZE))

    for page in range(1, pages + 1):
        url = f"{BASE_URL}/leagues-classic/{OVERALL_LEAGUE_ID}/standings/"
        try:
            r = session.get(url, params={"page_standings": page}, timeout=10)
            r.raise_for_status()
            data = r.json()
            results = data.get('standings', {}).get('results', [])
            if not results:
                break
            entries.extend(res['entry'] for res in results)
            has_next = data.get('standings', {}).get('has_next', False)
        except Exception:
            has_next = True  # on retente la page suivante malgré l'échec

        if progress_callback:
            progress_callback(page, pages)

        time.sleep(REQUEST_SLEEP)

        if not has_next:
            break

    return entries[:n]


def fetch_manager_picks(entry_id, gameweek, session=None):
    """Récupère les picks d'un manager pour une GW donnée. Renvoie None en cas d'échec."""
    session = session or _build_session()
    url = f"{BASE_URL}/entry/{entry_id}/event/{gameweek}/picks/"
    try:
        r = session.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fetch_player_mapping(session=None):
    """Mappe player_id -> {name, team, position}."""
    session = session or _build_session()
    url = f"{BASE_URL}/bootstrap-static/"
    r = session.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    teams = {t['id']: t['name'] for t in data['teams']}
    mapping = {}
    for p in data['elements']:
        mapping[p['id']] = {
            'name': p['web_name'],
            'team': teams.get(p['team'], '?'),
            'position': POS_MAP.get(p['element_type'], '?'),
        }
    return mapping


def _enrich(pid, count, denom, player_mapping):
    info = player_mapping.get(pid, {'name': f"ID {pid}", 'team': '?', 'position': '?'})
    return {
        'player_id': pid,
        'name': info['name'],
        'team': info['team'],
        'position': info['position'],
        'count': count,
        'pct': round(100 * count / denom, 1) if denom else 0.0,
    }


def build_consensus(gameweek, n=1000, force_refresh=False, progress_callback=None, session=None):
    """Orchestre le scraping et produit le consensus élite pour une GW.

    progress_callback(fraction: float 0..1, label: str) est appelé régulièrement
    pour piloter une barre de progression (ex: st.progress).
    """
    if not force_refresh:
        cached = load_cache(gameweek)
        if cached is not None:
            return cached

    session = session or _build_session()

    if progress_callback:
        progress_callback(0.0, "Chargement de la table des joueurs...")
    player_mapping = fetch_player_mapping(session=session)

    if progress_callback:
        progress_callback(0.02, "Récupération du top managers...")
    entries = fetch_top_managers(
        n, session=session,
        progress_callback=(lambda p, t: progress_callback(0.02 + 0.18 * p / t, f"Classement — page {p}/{t}"))
        if progress_callback else None,
    )

    captain_counts = {}
    ownership_counts = {}
    transfer_in_counts = {}
    n_valid = 0
    total = len(entries) if entries else 1

    for i, entry_id in enumerate(entries):
        picks_data = fetch_manager_picks(entry_id, gameweek, session=session)
        time.sleep(REQUEST_SLEEP)

        if not picks_data or 'picks' not in picks_data:
            if progress_callback:
                progress_callback(0.2 + 0.8 * (i + 1) / total, f"Analyse managers {i + 1}/{total}")
            continue

        n_valid += 1
        current_ids = set()
        for pick in picks_data['picks']:
            pid = pick['element']
            current_ids.add(pid)
            ownership_counts[pid] = ownership_counts.get(pid, 0) + 1
            if pick.get('is_captain'):
                captain_counts[pid] = captain_counts.get(pid, 0) + 1

        if gameweek > 1:
            prev_data = fetch_manager_picks(entry_id, gameweek - 1, session=session)
            time.sleep(REQUEST_SLEEP)
            if prev_data and 'picks' in prev_data:
                prev_ids = {p['element'] for p in prev_data['picks']}
                for pid in (current_ids - prev_ids):
                    transfer_in_counts[pid] = transfer_in_counts.get(pid, 0) + 1

        if progress_callback:
            progress_callback(0.2 + 0.8 * (i + 1) / total, f"Analyse managers {i + 1}/{total}")

    captain_consensus = sorted(
        (_enrich(pid, c, n_valid, player_mapping) for pid, c in captain_counts.items()),
        key=lambda x: x['count'], reverse=True
    )[:15]

    ownership_consensus = sorted(
        (_enrich(pid, c, n_valid, player_mapping)
         for pid, c in ownership_counts.items() if n_valid and c / n_valid >= 0.5),
        key=lambda x: x['count'], reverse=True
    )

    transfer_trends = sorted(
        (_enrich(pid, c, n_valid, player_mapping) for pid, c in transfer_in_counts.items()),
        key=lambda x: x['count'], reverse=True
    )[:20]

    result = {
        'gameweek': gameweek,
        'n_requested': n,
        'n_valid_managers': n_valid,
        'captain_consensus': captain_consensus,
        'ownership_consensus': ownership_consensus,
        'transfer_trends': transfer_trends,
    }

    save_cache(gameweek, result)
    return result
