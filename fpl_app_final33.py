import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import numpy as np

st.set_page_config(
    page_title="FPL Decision Tool",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Owner shortcut : entrer 1 charge directement cette team ──
OWNER_ID = 2999747

# --- 0. CONFIGURATION RESEAU ROBUSTE (Anti-Crash) ---
def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    return session

http = create_session()

# --- 1. FONCTIONS DE CHARGEMENT ET CALCULS ---

@st.cache_data(ttl=3600)
def get_player_recent_data(current_gw):
    player_data = {}
    for gw in range(current_gw, current_gw - 4, -1):
        if gw < 1: continue
        url = f"https://fantasy.premierleague.com/api/event/{gw}/live/"
        try:
            data = http.get(url, timeout=10).json()
            for p in data['elements']:
                p_id = p['id']
                s = p['stats']
                if p_id not in player_data: player_data[p_id] = []
                
                cbi = int(s.get('clearances_blocks_interceptions', 0))
                tackles = int(s.get('tackles', 0))
                cbit_match = cbi + tackles
                
                player_data[p_id].append({
                    'gw': gw,
                    'xg': float(s.get('expected_goals', 0)),
                    'xa': float(s.get('expected_assists', 0)),
                    'xgc': float(s.get('expected_goals_conceded', 0)),
                    'cbit': cbit_match,
                    'mins': int(s.get('minutes', 0))
                })
        except Exception as e:
            continue
    return player_data

@st.cache_data(ttl=3600)
def load_all_fpl_data():
    base_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        base_data = http.get(base_url, timeout=10).json()
    except:
        st.error("Erreur de connexion API FPL.")
        return pd.DataFrame(), pd.DataFrame()
    
    df_teams_stats = pd.DataFrame([{
        'Equipe': t['name'],
        'Att_H': t['strength_attack_home'], 'Att_A': t['strength_attack_away'],
        'Def_H': t['strength_defence_home'], 'Def_A': t['strength_defence_away']
    } for t in base_data['teams']])
    
    teams_dict = {t['id']: {'name': t['name'], 
                            'att_h': t['strength_attack_home'], 'att_a': t['strength_attack_away'],
                            'def_h': t['strength_defence_home'], 'def_a': t['strength_defence_away']} 
                  for t in base_data['teams']}
    
    try:
        fix_data = http.get("https://fantasy.premierleague.com/api/fixtures/", timeout=10).json()
    except:
        fix_data = []

    current_gw = next(
        (e['id'] for e in base_data['events'] if e['is_current']),
        next((e['id'] for e in base_data['events'] if e['is_next']), 1)
    )
    recent_stats = get_player_recent_data(current_gw)
    
    # --- ETAPE 1 : xGC DE RÉFÉRENCE ---
    team_reference_xgc = {} 
    temp_team_defenders = {} 
    
    for p in base_data['elements']:
        if p['element_type'] == 2: 
            t_id = p['team']
            matches = recent_stats.get(p['id'], [])
            total_mins = sum(m['mins'] for m in matches)
            total_xgc = sum(m['xgc'] for m in matches)
            if t_id not in temp_team_defenders: temp_team_defenders[t_id] = []
            temp_team_defenders[t_id].append({'mins': total_mins, 'xgc': total_xgc})
    
    for t_id, defs in temp_team_defenders.items():
        if not defs: continue
        boss = max(defs, key=lambda x: x['mins'])
        team_reference_xgc[t_id] = boss['xgc']

    # --- ETAPE 2 : BOUCLE PRINCIPALE ---
    player_list = []
    
    for p in base_data['elements']:
        t_id = p['team']
        pos_id = p['element_type']
        pos_name = {1:'GKP', 2:'DEF', 3:'MID', 4:'FWD'}[pos_id]
        matches = recent_stats.get(p['id'], [])
        sum_mins = sum(m['mins'] for m in matches)
        
        # N4DS
        upcoming = [f for f in fix_data if not f['finished'] and (f['team_h'] == t_id or f['team_a'] == t_id)][:4]
        total_difficulty = 0
        match_details = []
        for f in upcoming:
            is_home = (f['team_h'] == t_id)
            opp_id = f['team_a'] if is_home else f['team_h']
            if pos_name == 'DEF' or pos_name == 'GKP':
                val = teams_dict[opp_id]['att_a'] if is_home else teams_dict[opp_id]['att_h']
                label = "Att"
            else:
                val = teams_dict[opp_id]['def_a'] if is_home else teams_dict[opp_id]['def_h']
                label = "Def"
            total_difficulty += val
            match_details.append(f"{teams_dict[opp_id]['name']} ({'H' if is_home else 'A'}) | {label}: {val}")
        n4ds_val = round(total_difficulty / 4, 1) if upcoming else 0

        # Offensif
        sum_xg = sum(m['xg'] for m in matches)
        sum_xa = sum(m['xa'] for m in matches)
        score_off = (2 * sum_xg) + sum_xa if pos_name == 'FWD' else sum_xg + sum_xa
        projected_off = round((score_off / 4) * 3, 2)
        sum_off_pure = sum_xg + sum_xa

        # Defensif
        xgc_ref = team_reference_xgc.get(t_id, 0)
        sum_cbit_4 = sum(m['cbit'] for m in matches)
        
        if sum_mins >= 270: phase = "1. Titulaire Fixe"
        elif sum_mins >= 180: phase = "2. Temps élevé"
        elif sum_mins >= 90: phase = "3. Rotation"
        else: phase = "4. Faible"

        joueur_data = {
            "ID": p['id'],                          # ← AJOUT pour liaison My Team
            "Joueur": p['web_name'], 
            "Equipe": teams_dict[t_id]['name'], 
            "Position": pos_name,
            "Prix": p['now_cost'] / 10,             # ← AJOUT prix en M£
            "N4DS": n4ds_val, 
            "Score_Off": projected_off, 
            "Achetés": p['transfers_in_event'], 
            "Régularité": phase, 
            "Détails_N4DS": match_details, 
            "Détails_Matches": matches,
            "Stats_Def_Hybrides": {
                'xGC_Team': xgc_ref, 
                'DC_Indiv': sum_cbit_4,
                'Off_Indiv': sum_off_pure
            },
            "XGCDC": 0
        }
        player_list.append(joueur_data)
    
    # --- ETAPE 3 : NORMALISATION (3 FACTEURS) ---
    df = pd.DataFrame(player_list)
    def_mask = (df['Position'] == 'DEF') | (df['Position'] == 'GKP')
    
    if not df.empty and def_mask.any():
        xgc_vals = df.loc[def_mask, 'Stats_Def_Hybrides'].apply(lambda x: x['xGC_Team'])
        dc_vals = df.loc[def_mask, 'Stats_Def_Hybrides'].apply(lambda x: x['DC_Indiv'])
        off_vals = df.loc[def_mask, 'Stats_Def_Hybrides'].apply(lambda x: x['Off_Indiv'])
        
        max_xgc, min_xgc = xgc_vals.max(), xgc_vals.min()
        max_dc, min_dc = dc_vals.max(), dc_vals.min()
        max_off, min_off = off_vals.max(), off_vals.min()
        
        def calculate_xgcdc_row(row):
            if row['Position'] not in ['DEF', 'GKP']: return 0
            stats = row['Stats_Def_Hybrides']
            norm_xgc = (max_xgc - stats['xGC_Team']) / (max_xgc - min_xgc) if max_xgc != min_xgc else 0
            norm_dc = (stats['DC_Indiv'] - min_dc) / (max_dc - min_dc) if max_dc != min_dc else 0
            norm_off = (stats['Off_Indiv'] - min_off) / (max_off - min_off) if max_off != min_off else 0
            raw_score = (norm_xgc * 0.50) + (norm_dc * 0.15) + (norm_off * 0.10)
            final_score = (raw_score / 0.75) * 100
            return round(final_score, 1)

        df['XGCDC'] = df.apply(calculate_xgcdc_row, axis=1)
    return df_teams_stats, df

df_teams_raw, df_players = load_all_fpl_data()

# ============================================================
# --- NOUVELLE FONCTION : CHARGEMENT MA TEAM ---
# ============================================================

@st.cache_data(ttl=300)
def load_my_team(team_id):
    try:
        current_gw = next(
            (e['id'] for e in http.get(
                "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10
            ).json()['events'] if e['is_current']),
            1
        )

        info = http.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/",
            timeout=10
        ).json()

        history = http.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/history/",
            timeout=10
        ).json()

        picks = http.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{current_gw}/picks/",
            timeout=10
        ).json()

        transfers = http.get(
            f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/",
            timeout=10
        ).json()

        return info, history, picks, transfers, current_gw

    except Exception as e:
        return None, None, None, None, None


def enrich_picks_with_scores(picks_data, df_players):
    picks = picks_data.get('picks', [])
    entry_history = picks_data.get('entry_history', {})
    id_to_row = df_players.set_index('ID')

    enriched = []
    for pick in picks:
        pid = pick['element']
        if pid in id_to_row.index:
            p = id_to_row.loc[pid]
            captain_label = "©" if pick['is_captain'] else ("V©" if pick['is_vice_captain'] else "")
            enriched.append({
                'Joueur': f"{p['Joueur']} {captain_label}".strip(),
                'Equipe': p['Equipe'],
                'Pos': p['Position'],
                'Prix': p['Prix'],
                'Score_Off': p['Score_Off'],
                'XGCDC': p['XGCDC'],
                'N4DS': p['N4DS'],
                'Régularité': p['Régularité'],
                'Capitaine': pick['is_captain'],
                'Vice': pick['is_vice_captain'],
                'Titulaire': pick['position'] <= 11,
            })

    return pd.DataFrame(enriched), entry_history


# ============================================================
# --- 2. PREDICTIONS ML ---
# ============================================================

@st.cache_data(ttl=3600)
def load_predictions():
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None, "scikit-learn non installé. Lancez : pip install scikit-learn"

    try:
        base_data = http.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
    except Exception as e:
        return None, f"Erreur API bootstrap : {e}"

    try:
        fix_data = http.get("https://fantasy.premierleague.com/api/fixtures/", timeout=10).json()
    except Exception:
        fix_data = []

    teams_dict = {t['id']: t['name'] for t in base_data['teams']}

    # Build team → [fdr_gw1, fdr_gw2] depuis les prochains matchs
    upcoming = [f for f in fix_data if not f.get('finished', True)]
    team_fdrs = {}
    for f in upcoming:
        for side, diff_key in [('team_h', 'team_h_difficulty'), ('team_a', 'team_a_difficulty')]:
            tid = f[side]
            if tid not in team_fdrs:
                team_fdrs[tid] = []
            if len(team_fdrs[tid]) < 2:
                team_fdrs[tid].append(int(f.get(diff_key) or 3))

    def _sf(val):
        try:
            return float(val or 0)
        except (TypeError, ValueError):
            return 0.0

    rows = []
    for p in base_data['elements']:
        tid = p['team']
        fdrs = team_fdrs.get(tid, [])
        fdr1 = fdrs[0] if len(fdrs) > 0 else 3
        fdr2 = fdrs[1] if len(fdrs) > 1 else fdr1

        rows.append({
            'Joueur':       p['web_name'],
            'Equipe':       teams_dict.get(tid, ''),
            'Position':     {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}[p['element_type']],
            'pos_id':       p['element_type'],
            'Prix':         p['now_cost'] / 10,
            'form':         _sf(p.get('form')),
            'minutes':      int(p.get('minutes', 0)),
            'total_points': int(p.get('total_points', 0)),
            'ict_index':    _sf(p.get('ict_index')),
            'fdr_gw1':      fdr1,
            'fdr_gw2':      fdr2,
            'ep_next':      _sf(p.get('ep_next')),
        })

    df_ml = pd.DataFrame(rows)

    FEATURES = ['form', 'minutes', 'total_points', 'Prix', 'ict_index', 'pos_id', 'fdr']
    df_train = df_ml[df_ml['minutes'] >= 90].copy()

    if len(df_train) < 30:
        return None, "Données insuffisantes pour entraîner le modèle (< 30 joueurs avec minutes >= 90)."

    df_train = df_train.copy()
    df_train['fdr'] = df_train['fdr_gw1']
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(df_train[FEATURES].values, df_train['ep_next'].values)

    # Prédiction GW+1
    df_ml['fdr'] = df_ml['fdr_gw1']
    df_ml['xP_GW1'] = model.predict(df_ml[FEATURES].values).clip(0).round(2)

    # Prédiction GW+2 (même modèle, FDR du match suivant)
    df_ml['fdr'] = df_ml['fdr_gw2']
    df_ml['xP_GW2'] = model.predict(df_ml[FEATURES].values).clip(0).round(2)

    df_ml['xP_Total'] = (df_ml['xP_GW1'] + df_ml['xP_GW2']).round(2)
    df_ml.drop(columns=['fdr'], inplace=True)

    return df_ml, None


# ============================================================
# --- NEW : PRÉDICTIONS V6 (modèles pkl pré-entraînés) ---
# ============================================================

@st.cache_data(ttl=3600)
def get_xp_predictions():
    # 1. Charge les 4 modèles V6
    models = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        path = f'model_v6_{pos}.pkl'
        if os.path.exists(path):
            try:
                with open(path, 'rb') as _f:
                    models[pos] = pickle.load(_f)
            except Exception:
                pass

    if not models:
        return None, "Aucun modèle V6 trouvé. Placez les fichiers model_v6_*.pkl dans le dossier de l'app."

    # 2. Charge les features SHAP
    top_features = {}
    if os.path.exists('shap_v6_top_features.pkl'):
        try:
            with open('shap_v6_top_features.pkl', 'rb') as _f:
                top_features = pickle.load(_f)
        except Exception:
            pass

    # 3. Charge les caches optionnels
    cache_elo = {}
    if os.path.exists('cache_elo.pkl'):
        try:
            with open('cache_elo.pkl', 'rb') as _f:
                cache_elo = pickle.load(_f)
        except Exception:
            pass

    # 4. API FPL
    try:
        base_data = http.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
    except Exception as e:
        return None, f"Erreur API FPL : {e}"

    try:
        fix_data = http.get("https://fantasy.premierleague.com/api/fixtures/", timeout=10).json()
    except Exception:
        fix_data = []

    teams_dict = {t['id']: t['name'] for t in base_data['teams']}

    current_gw = next(
        (e['id'] for e in base_data['events'] if e['is_current']),
        next((e['id'] for e in base_data['events'] if e['is_next']), 1)
    )

    upcoming_fixes = [f for f in fix_data if not f.get('finished', True)]

    # Prochain event GW
    next_events = sorted({f.get('event') for f in upcoming_fixes if f.get('event')})
    next_event_id = next_events[0] if next_events else current_gw + 1

    # Fixtures par équipe (max 2)
    team_next_fixes = {}
    for f in upcoming_fixes:
        for side in ['team_h', 'team_a']:
            tid = f[side]
            if tid not in team_next_fixes:
                team_next_fixes[tid] = []
            if len(team_next_fixes[tid]) < 2:
                team_next_fixes[tid].append(f)

    # Détection DGW
    team_next_event_count = {}
    for f in upcoming_fixes:
        if f.get('event') == next_event_id:
            for side in ['team_h', 'team_a']:
                tid = f[side]
                team_next_event_count[tid] = team_next_event_count.get(tid, 0) + 1

    def _sf(val):
        try:
            return float(val or 0)
        except Exception:
            return 0.0

    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    records = []
    for p in base_data['elements']:
        tid = p['team']
        pos_id = p['element_type']
        pos = pos_map.get(pos_id, 'MID')

        fixes = team_next_fixes.get(tid, [])
        nf = fixes[0] if fixes else None
        was_home = bool(nf and nf['team_h'] == tid)

        fdr_gw1 = 3
        if nf:
            fdr_key = 'team_h_difficulty' if nf['team_h'] == tid else 'team_a_difficulty'
            fdr_gw1 = int(nf.get(fdr_key) or 3)

        opp_name = "?"
        if nf:
            opp_tid = nf['team_a'] if was_home else nf['team_h']
            loc = "H" if was_home else "A"
            opp_name = f"{teams_dict.get(opp_tid, '?')} ({loc})"

        games_next_gw = team_next_event_count.get(tid, 1)
        is_dgw = games_next_gw >= 2

        chance_raw = p.get('chance_of_playing_next_round')
        chance_pct = float(chance_raw) if chance_raw is not None else 100.0
        chance_norm = chance_pct / 100.0

        mins = int(p.get('minutes', 0))
        exp_mins = (mins / max(current_gw, 1)) * chance_norm * games_next_gw

        # ELO
        elo_win_prob = 0.5
        if isinstance(cache_elo, dict):
            elo_v = cache_elo.get(tid, cache_elo.get(str(tid)))
            if isinstance(elo_v, dict):
                elo_win_prob = float(elo_v.get('win_prob', elo_v.get('prob', 0.5)))
            elif isinstance(elo_v, (int, float)):
                elo_win_prob = float(elo_v) if 0.0 <= float(elo_v) <= 1.0 else 0.5

        feat = {
            'minutes': float(mins),
            'value': float(p.get('now_cost', 0)),
            'now_cost': p.get('now_cost', 0) / 10,
            'ict_index': _sf(p.get('ict_index')),
            'influence': _sf(p.get('influence')),
            'selected_by_percent': _sf(p.get('selected_by_percent')),
            'transfers_in': float(p.get('transfers_in_event', 0)),
            'transfers_out': float(p.get('transfers_out_event', 0)),
            'transfers_in_event': float(p.get('transfers_in_event', 0)),
            'transfers_out_event': float(p.get('transfers_out_event', 0)),
            'creativity': _sf(p.get('creativity')),
            'threat': _sf(p.get('threat')),
            'bps': float(p.get('bps', 0)),
            'form': _sf(p.get('form')),
            'was_home': float(was_home),
            'chance_of_playing_next_round': chance_pct,
            'chance_of_playing': chance_norm,
            'games_next_gw': float(games_next_gw),
            'expected_minutes': exp_mins,
            'fdr': float(fdr_gw1),
            'fdr_gw1': float(fdr_gw1),
            'total_points': float(p.get('total_points', 0)),
            'ep_next': _sf(p.get('ep_next')),
            'ep_this': _sf(p.get('ep_this')),
            'pos_id': float(pos_id),
            'element_type': float(pos_id),
            'elo_win_prob': elo_win_prob,
        }

        records.append({
            '_feat': feat,
            'id': p['id'],
            'name': p['web_name'],
            'team': teams_dict.get(tid, ''),
            'team_id': tid,
            'position': pos,
            'price': p.get('now_cost', 0) / 10,
            'chance_of_playing': int(chance_pct),
            'is_double_gw': is_dgw,
            'form': _sf(p.get('form')),
            'fdr_next': fdr_gw1,
            'opp_name': opp_name,
            'was_home': was_home,
            'elo_win_prob': elo_win_prob,
        })

    # 6. Prédictions
    df_rows = []
    for rec in records:
        pos = rec['position']
        mdl = models.get(pos)
        xp_gw1 = 0.0

        if mdl is not None:
            if isinstance(top_features, dict) and pos in top_features:
                feat_names = list(top_features[pos])
            elif isinstance(top_features, list) and top_features:
                feat_names = list(top_features)
            elif hasattr(mdl, 'feature_names_in_'):
                feat_names = list(mdl.feature_names_in_)
            elif hasattr(mdl, 'feature_names'):
                feat_names = list(mdl.feature_names)
            else:
                feat_names = list(rec['_feat'].keys())

            fvec = np.array([[float(rec['_feat'].get(fn, 0)) for fn in feat_names]])
            try:
                xp_gw1 = float(mdl.predict(fvec)[0])
                xp_gw1 = max(0.0, xp_gw1)
            except Exception:
                xp_gw1 = 0.0

        xp_gw2 = round(xp_gw1 * 0.95, 2)
        df_rows.append({
            'id': rec['id'],
            'name': rec['name'],
            'team': rec['team'],
            'team_id': rec['team_id'],
            'position': rec['position'],
            'price': rec['price'],
            'xP_GW1': round(xp_gw1, 2),
            'xP_GW2': xp_gw2,
            'xP_Total': round(xp_gw1 + xp_gw2, 2),
            'chance_of_playing': rec['chance_of_playing'],
            'is_double_gw': rec['is_double_gw'],
            'form': rec['form'],
            'fdr_next': rec['fdr_next'],
            'opp_name': rec['opp_name'],
            'was_home': rec['was_home'],
            'elo_win_prob': rec['elo_win_prob'],
        })

    return pd.DataFrame(df_rows), None


# ============================================================
# --- 3. SIDEBAR ---
# ============================================================

st.sidebar.title("🎮 FPL Command Center")

# ── NAVIGATION PRINCIPALE ──
st.sidebar.markdown("### 📌 Navigation")
page = st.sidebar.radio(
    label="",
    options=[
        "📊 Analyse Stratégique",
        "👤 Ma Team",
        "🔮 Prédictions xP",
        "🧪 Prédictions V6",
        "🔄 Transferts",
        "⚽ Capitaine",
    ],
    index=0
)

if st.sidebar.button("🔄 Rafraîchir les données"):
    st.cache_data.clear()
    st.rerun()

# ── Contenu sidebar selon la page ──
if page == "📊 Analyse Stratégique":
    with st.sidebar.expander("📊 Forces Équipes (H/A)"):
        if not df_teams_raw.empty:
            st.dataframe(df_teams_raw, hide_index=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Localiser un Joueur")
    if not df_players.empty:
        all_player_names = sorted(df_players['Joueur'].unique())
        search_player = st.sidebar.selectbox("Choisissez un nom :", [""] + all_player_names)
    else:
        search_player = ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Choix des Diagrammes")
    show_fwd = st.sidebar.checkbox("Attaquants (FWD)", value=True)
    show_mid = st.sidebar.checkbox("Milieux (MID)", value=True)
    show_def = st.sidebar.checkbox("Défenseurs (DEF)", value=True)

elif page == "👤 Ma Team":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🆔 Ton ID FPL")
    _team_id_raw = st.sidebar.number_input(
        "Entre ton Team ID :",
        min_value=1, max_value=99999999,
        value=1, step=1, format="%d",
        help="Entre 1 pour accès owner direct"
    )
    team_id_input = OWNER_ID if int(_team_id_raw) == 1 else int(_team_id_raw)
    if team_id_input == OWNER_ID:
        st.sidebar.success("👑 Owner connecté")
    else:
        st.sidebar.info("💡 Trouve ton ID sur : fantasy.premierleague.com → Mon Équipe → URL")

elif page == "🔮 Prédictions xP":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Filtres Prédictions")
    pred_pos_filter = st.sidebar.selectbox(
        "Position",
        ["Tous", "GKP", "DEF", "MID", "FWD"],
        help="Filtrer par poste"
    )
    pred_search = st.sidebar.text_input(
        "🔍 Rechercher un joueur",
        placeholder="ex: Salah"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Modèle : RandomForest (scikit-learn)\n"
        "Features : form · minutes · total_points\n"
        "prix · ICT index · position · FDR\n"
        "Target : ep_next (FPL)"
    )

elif page == "🧪 Prédictions V6":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Filtres V6")
    v6_pos_sidebar = st.sidebar.selectbox(
        "Position", ["Tous", "GK", "DEF", "MID", "FWD"], key="sb_v6_pos"
    )
    v6_price_sidebar = st.sidebar.slider(
        "Prix max (£m)", 4.0, 16.0, 16.0, step=0.5, key="sb_v6_prix"
    )
    v6_search_sidebar = st.sidebar.text_input(
        "Rechercher", placeholder="ex: Salah", key="sb_v6_search"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Modèles : V6 pkl pré-entraînés\n"
        "GK · DEF · MID · FWD\n"
        "Features : SHAP top features\n"
        "xP_GW2 = xP_GW1 × 0.95"
    )

elif page == "🔄 Transferts":
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Entre ton Team ID FPL dans la page principale "
        "pour obtenir des suggestions de transferts basées "
        "sur les prédictions V6."
    )

elif page == "⚽ Capitaine":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚽ Score Capitaine")
    st.sidebar.caption(
        "Score = xP_GW1 × 2\n"
        "× elo_win_prob\n"
        "× chance_of_playing\n"
        "× (1 / FDR_next)"
    )


# ============================================================
# --- 3. PAGE : ANALYSE STRATÉGIQUE (code original intact) ---
# ============================================================

if page == "📊 Analyse Stratégique":

    color_map = {"1. Titulaire Fixe": "green", "2. Temps élevé": "blue", "3. Rotation": "yellow", "4. Faible": "red"}
    selected_player_data = None
    selected_chart_type = None

    def draw_chart_interactive(df, x_col, y_col, title, x_label, y_label, key_name, highlight_name=None):
        st.subheader(title)
        if df.empty:
            st.warning("Aucune donnée disponible.")
            return None

        fig = px.scatter(df, x=x_col, y=y_col, color="Régularité", color_discrete_map=color_map,
                         hover_name="Joueur", size="Achetés", custom_data=["Joueur"])
        
        if highlight_name and highlight_name in df['Joueur'].values:
            target = df[df['Joueur'] == highlight_name].iloc[0]
            fig.add_trace(go.Scatter(
                x=[target[x_col]], y=[target[y_col]],
                mode='markers',
                marker=dict(color='red', size=20, symbol='circle-open', line=dict(width=3)),
                name=highlight_name, hoverinfo='skip'
            ))
            fig.add_annotation(
                x=target[x_col], y=target[y_col], text=f"📍 {highlight_name}",
                showarrow=True, arrowhead=2, ax=0, ay=-40,
                bgcolor="white", bordercolor="red"
            )

        fig.update_xaxes(autorange="reversed", title_text=x_label)
        fig.update_yaxes(title_text=y_label)
        
        selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=key_name)
        if selection and selection["selection"]["points"]:
            return selection["selection"]["points"][0]["customdata"][0]
        return None

    st.title("⚽ Analyse Stratégique FPL Ultimate")

    if show_fwd and not df_players.empty:
        df_fwd = df_players[df_players['Position'] == 'FWD']
        sel_fwd = draw_chart_interactive(df_fwd, "N4DS", "Score_Off", "🚀 ATTAQUANTS", "Difficulté Défense (N4DS)", "Score XAG", "chart_fwd", search_player)
        if sel_fwd: 
            selected_player_data = df_fwd[df_fwd['Joueur'] == sel_fwd].iloc[0]
            selected_chart_type = "ATT"

    if show_mid and not df_players.empty:
        st.markdown("---")
        df_mid = df_players[df_players['Position'] == 'MID']
        sel_mid = draw_chart_interactive(df_mid, "N4DS", "Score_Off", "🎯 MILIEUX", "Difficulté Défense (N4DS)", "Score XGA", "chart_mid", search_player)
        if sel_mid: 
            selected_player_data = df_mid[df_mid['Joueur'] == sel_mid].iloc[0]
            selected_chart_type = "ATT"

    if show_def and not df_players.empty:
        st.markdown("---")
        df_def = df_players[df_players['Position'] == 'DEF']
        sel_def = draw_chart_interactive(df_def, "N4DS", "XGCDC", "🛡️ DÉFENSEURS", "Difficulté Attaque (N4DS)", "Index XGCDC (3 Facteurs)", "chart_def", search_player)
        if sel_def: 
            selected_player_data = df_def[df_def['Joueur'] == sel_def].iloc[0]
            selected_chart_type = "DEF"

    # --- INSPECTEUR ---
    st.markdown("---")
    if selected_player_data is not None:
        p = selected_player_data
        st.header(f"🔍 Analyse : {p['Joueur']} ({p['Equipe']})")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📅 N4DS (Calendrier)")
            if selected_chart_type == "DEF": st.info("Défenseur vs ATTAQUE adverse.")
            else: st.info("Attaquant vs DÉFENSE adverse.")
            for m in p['Détails_N4DS']: st.write(f"- {m}")
            st.write(f"👉 **Moyenne N4DS = {p['N4DS']}**")

        with col2:
            if selected_chart_type == "DEF":
                st.subheader("🛡️ Index XGCDC (3 Facteurs)")
                stats = p['Stats_Def_Hybrides']
                st.write(f"**1. Solidité Équipe (50%) :**")
                st.write(f"- xGC Équipe (Ref) : **{round(stats['xGC_Team'], 2)}**")
                st.write(f"**2. Activité Individuelle (15%) :**")
                st.write(f"- Actions (CBIT) : **{stats['DC_Indiv']}**")
                st.write(f"**3. Potentiel Offensif (10%) :**")
                st.write(f"- xG + xA : **{round(stats['Off_Indiv'], 2)}**")
                st.latex(r'''Index = \frac{(Norm_{xGC} \times 0.50) + (Norm_{DC} \times 0.15) + (Norm_{Off} \times 0.10)}{0.75}''')
                st.success(f"Index XGCDC = **{p['XGCDC']} / 100**")
            else:
                st.subheader("🚀 Score Offensif")
                st.dataframe(pd.DataFrame(p['Détails_Matches'])[['gw', 'xg', 'xa', 'mins']], hide_index=True)
                st.success(f"Score Projeté = **{p['Score_Off']}**")
    else:
        st.info("👆 **Cliquez sur un joueur (ou utilisez la recherche) pour voir les détails.**")


# ============================================================
# --- 4. PAGE : MA TEAM V2 ---
# ============================================================

elif page == "👤 Ma Team":

    st.title("👤 Ma Team FPL")

    if team_id_input == OWNER_ID:
        st.caption(f"👑 Owner — Team ID : {OWNER_ID}")
    else:
        st.caption(f"Team ID : {team_id_input}")

    with st.spinner(f"Chargement de la team {team_id_input}..."):
        info, history, picks_data, transfers, current_gw = load_my_team(team_id_input)

        if info is None:
            st.error("❌ Team ID introuvable. Vérifie le numéro et réessaie.")
            st.stop()

        # ── 1. HEADER + KPIs ──
        st.header(f"🏟️ {info.get('name', 'Mon Équipe')}  —  {info.get('player_first_name', '')} {info.get('player_last_name', '')}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 Points Total", f"{info.get('summary_overall_points', 0):,}")
        k2.metric("🌍 Rank Global", f"#{info.get('summary_overall_rank', 0):,}")
        k3.metric(f"⚡ Points GW{current_gw}", info.get('summary_event_points', 0))
        k4.metric("💰 Valeur Équipe", f"{info.get('last_deadline_value', 0) / 10:.1f} M£")

        st.markdown("---")

        # ── 2. ÉVOLUTION RANK GLOBAL ──
        if history and history.get('current'):
            df_hist = pd.DataFrame(history['current'])
            fig_rank = px.line(
                df_hist, x='event', y='overall_rank',
                title="📈 Évolution Rank Global",
                labels={'event': 'Gameweek', 'overall_rank': 'Rank'},
                markers=True
            )
            fig_rank.update_yaxes(autorange="reversed")
            fig_rank.update_traces(line_color='#1f77b4', marker_color='#1f77b4')
            st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("---")

        # ── Chargement squad + xP V6 ──
        if picks_data and not df_players.empty:
            df_my_team, entry_hist = enrich_picks_with_scores(picks_data, df_players)

            b1, b2, b3 = st.columns(3)
            b1.info(f"💰 Valeur équipe : **{entry_hist.get('value', 0) / 10:.1f} M£**")
            b2.info(f"🏦 En banque : **{entry_hist.get('bank', 0) / 10:.1f} M£**")
            b3.info(f"⚡ Points cette GW : **{entry_hist.get('points', 0)} pts**")

            if not df_my_team.empty:

                # Enrichissement xP V6
                def _strip(n):
                    return str(n).replace(" ©", "").replace(" V©", "").strip()

                df_v6_mt = None
                try:
                    df_v6_mt, _ = get_xp_predictions()
                except Exception:
                    pass

                if df_v6_mt is not None and not df_v6_mt.empty:
                    _v6idx = df_v6_mt.set_index('name')[
                        ['xP_GW1', 'xP_GW2', 'chance_of_playing', 'fdr_next', 'opp_name']
                    ].to_dict('index')
                    df_my_team['xP+1']   = df_my_team['Joueur'].apply(lambda n: _v6idx.get(_strip(n), {}).get('xP_GW1', None))
                    df_my_team['xP+2']   = df_my_team['Joueur'].apply(lambda n: _v6idx.get(_strip(n), {}).get('xP_GW2', None))
                    df_my_team['_chance'] = df_my_team['Joueur'].apply(lambda n: _v6idx.get(_strip(n), {}).get('chance_of_playing', 100))
                    df_my_team['_fdr']   = df_my_team['Joueur'].apply(lambda n: _v6idx.get(_strip(n), {}).get('fdr_next', 3))
                    df_my_team['_opp']   = df_my_team['Joueur'].apply(lambda n: _v6idx.get(_strip(n), {}).get('opp_name', '?'))
                else:
                    df_my_team['xP+1']   = None
                    df_my_team['xP+2']   = None
                    df_my_team['_chance'] = 100
                    df_my_team['_fdr']   = 3
                    df_my_team['_opp']   = '?'

                _reg_emoji = {
                    "1. Titulaire Fixe": "🟢", "2. Temps élevé": "🔵",
                    "3. Rotation": "🟡", "4. Faible": "🔴"
                }
                df_my_team['État'] = df_my_team['Régularité'].map(_reg_emoji).fillna("⚪")

                starters = df_my_team[df_my_team['Titulaire']].copy()
                bench    = df_my_team[~df_my_team['Titulaire']].copy()

                # ── 3. CAPITAINE SUGGÉRÉ ──
                st.subheader("⚽ Capitaine suggéré — cette GW")
                _starters_xp = starters[starters['xP+1'].notna()].copy()
                if not _starters_xp.empty:
                    _starters_xp['_cap_score'] = _starters_xp.apply(
                        lambda r: float(r['xP+1']) * 2.0 * (float(r['_chance']) / 100.0), axis=1
                    )
                    _cap_sorted = _starters_xp.sort_values('_cap_score', ascending=False).reset_index(drop=True)
                    _cap  = _cap_sorted.iloc[0]
                    _vice = _cap_sorted.iloc[1] if len(_cap_sorted) > 1 else None

                    _col_cap, _col_vice = st.columns(2)
                    with _col_cap:
                        st.success(
                            f"**🥇 {_strip(_cap['Joueur'])}**\n\n"
                            f"xP × 2 : **{float(_cap['xP+1']) * 2:.1f} pts**\n\n"
                            f"{_cap['Equipe']} vs {_cap['_opp']} — FDR : {_cap['_fdr']}\n\n"
                            f"Dispo : {int(float(_cap['_chance']))}%"
                        )
                    if _vice is not None:
                        with _col_vice:
                            st.info(
                                f"**🥈 {_strip(_vice['Joueur'])}**\n\n"
                                f"xP × 2 : **{float(_vice['xP+1']) * 2:.1f} pts**\n\n"
                                f"{_vice['Equipe']} vs {_vice['_opp']} — FDR : {_vice['_fdr']}\n\n"
                                f"Dispo : {int(float(_vice['_chance']))}%"
                            )
                    st.caption("Basé sur xP V6 — parmi tes 11 titulaires")
                else:
                    st.info("xP V6 non disponible pour le calcul du capitaine.")

                st.markdown("---")

                # ── 4. PITCH VIEW ──
                try:
                    _pos_counts = starters['Pos'].value_counts()
                    _formation  = f"{int(_pos_counts.get('DEF',0))}-{int(_pos_counts.get('MID',0))}-{int(_pos_counts.get('FWD',0))}"

                    fig_pitch = go.Figure()

                    # Fond + bordure
                    fig_pitch.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                                        fillcolor="#1a5c2a", line=dict(color="white", width=2))
                    # Ligne médiane
                    fig_pitch.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                                        line=dict(color="white", width=1.5))
                    # Cercle central
                    fig_pitch.add_shape(type="circle", x0=41, y0=44, x1=59, y1=56,
                                        line=dict(color="white", width=1.5))
                    # Surfaces de réparation
                    fig_pitch.add_shape(type="rect", x0=18, y0=0, x1=82, y1=18,
                                        line=dict(color="white", width=1.5))
                    fig_pitch.add_shape(type="rect", x0=18, y0=82, x1=82, y1=100,
                                        line=dict(color="white", width=1.5))
                    # Buts
                    fig_pitch.add_shape(type="rect", x0=38, y0=0, x1=62, y1=5,
                                        line=dict(color="white", width=1.5))
                    fig_pitch.add_shape(type="rect", x0=38, y0=95, x1=62, y1=100,
                                        line=dict(color="white", width=1.5))

                    def _dot_color(xp_val):
                        if xp_val is None:
                            return '#888888'
                        v = float(xp_val)
                        return '#00ff87' if v >= 6.0 else ('#ffbe0b' if v >= 3.0 else '#ff006e')

                    _pos_y = {'GKP': 90, 'DEF': 70, 'MID': 48, 'FWD': 22}

                    for _pg in ['GKP', 'DEF', 'MID', 'FWD']:
                        _grp = starters[starters['Pos'] == _pg].reset_index(drop=True)
                        if _grp.empty:
                            continue
                        _n = len(_grp)
                        _yv = _pos_y[_pg]
                        _xs = [100 * (i + 1) / (_n + 1) for i in range(_n)]

                        for _ei, (_, _rp) in enumerate(_grp.iterrows()):
                            _xp1 = _rp['xP+1']
                            _xp2 = _rp['xP+2']
                            _dc  = _dot_color(_xp1)
                            _s1  = f"{float(_xp1):.1f}" if _xp1 is not None else "N/A"
                            _s2  = f"{float(_xp2):.1f}" if _xp2 is not None else "N/A"
                            _is_cap = bool(_rp.get('Capitaine', False))
                            _sname  = _strip(_rp['Joueur']).split()[-1][:9]
                            _label  = f"{_sname} {'⭐' if _is_cap else ''}<br>{_s1}".strip()

                            fig_pitch.add_trace(go.Scatter(
                                x=[_xs[_ei]], y=[_yv],
                                mode='markers',
                                marker=dict(color=_dc, size=32,
                                            line=dict(color='white', width=2)),
                                name=_strip(_rp['Joueur']),
                                hovertemplate=(
                                    f"<b>{_strip(_rp['Joueur'])}</b><br>"
                                    f"{_rp['Equipe']}<br>"
                                    f"Prix : £{_rp['Prix']:.1f}m<br>"
                                    f"xP+1 : {_s1}<br>"
                                    f"xP+2 : {_s2}"
                                    "<extra></extra>"
                                ),
                                showlegend=False
                            ))
                            fig_pitch.add_annotation(
                                x=_xs[_ei], y=_yv - 8,
                                text=_label,
                                showarrow=False,
                                font=dict(color='white', size=9),
                                align='center'
                            )

                    fig_pitch.update_layout(
                        title=dict(
                            text=f"⚽ Formation {_formation} — GW{current_gw}",
                            font=dict(color='white', size=15)
                        ),
                        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(range=[-10, 108], showgrid=False, zeroline=False, visible=False),
                        height=580,
                        plot_bgcolor='#1a5c2a',
                        paper_bgcolor='#1a5c2a',
                        margin=dict(l=5, r=5, t=45, b=5),
                    )
                    st.plotly_chart(fig_pitch, use_container_width=True)

                    # Remplaçants sous le terrain
                    st.caption("🪑 Remplaçants")
                    _bcols = st.columns(len(bench))
                    for _bi, (_, _br) in enumerate(bench.reset_index(drop=True).iterrows()):
                        _bxp = _br['xP+1']
                        _bclr = "🟢" if (_bxp is not None and float(_bxp) >= 6.0) else \
                                ("🟡" if (_bxp is not None and float(_bxp) >= 3.0) else "🔴")
                        _bstr = f"{float(_bxp):.1f}" if _bxp is not None else "N/A"
                        _bcols[_bi].metric(
                            label=f"{_bclr} {_strip(_br['Joueur'])[:13]}",
                            value=f"xP: {_bstr}",
                            delta=f"£{_br['Prix']:.1f}m"
                        )

                except Exception as _e_pitch:
                    st.warning(f"Pitch View indisponible : {_e_pitch}")

                st.markdown("---")

                # ── 5 & 6. TABLEAUX TITULAIRES + REMPLAÇANTS ──
                st.subheader(f"⚽ Mon Équipe — GW{current_gw}")

                def _color_xp_cell(val):
                    try:
                        v = float(str(val))
                        if v >= 6.0:
                            return 'background-color: #d4edda; color: #155724'
                        if v >= 3.0:
                            return 'background-color: #fff3cd; color: #856404'
                        return 'background-color: #f8d7da; color: #721c24'
                    except (TypeError, ValueError):
                        return ''

                def _squad_table(df_sub):
                    t = df_sub[['Joueur', 'Pos', 'Equipe', 'Prix',
                                'xP+1', 'xP+2', 'Score_Off', 'N4DS', 'État']].copy()
                    t['xP+1'] = t['xP+1'].apply(lambda x: f"{float(x):.1f}" if x is not None else "N/A")
                    t['xP+2'] = t['xP+2'].apply(lambda x: f"{float(x):.1f}" if x is not None else "N/A")
                    return t

                st.write("**🟢 Titulaires (XI)**")
                try:
                    st.dataframe(
                        _squad_table(starters).style.map(_color_xp_cell, subset=['xP+1']),
                        hide_index=True, use_container_width=True
                    )
                except Exception:
                    st.dataframe(_squad_table(starters), hide_index=True, use_container_width=True)

                st.write("**🪑 Remplaçants**")
                try:
                    st.dataframe(
                        _squad_table(bench).style.map(_color_xp_cell, subset=['xP+1']),
                        hide_index=True, use_container_width=True
                    )
                except Exception:
                    st.dataframe(_squad_table(bench), hide_index=True, use_container_width=True)

                # ── 7. PROFIL ÉQUIPE ──
                st.markdown("---")
                st.subheader("📡 Profil de l'équipe")
                _pos_summary = df_my_team.groupby('Pos').agg(
                    Score_Off_Moy=('Score_Off', 'mean'),
                    XGCDC_Moy=('XGCDC', 'mean'),
                    N4DS_Moy=('N4DS', 'mean'),
                    Nb=('Joueur', 'count')
                ).reset_index()
                fig_radar = px.bar(
                    _pos_summary, x='Pos', y='Score_Off_Moy',
                    title="Score Offensif Moyen par Position",
                    color='Pos',
                    labels={'Score_Off_Moy': 'Score Off Moyen', 'Pos': 'Position'},
                    text_auto='.2f'
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")

        # ── 8. HISTORIQUE DES TRANSFERTS ──
        st.subheader("🔄 Historique des Transferts")

        if transfers:
            id_to_name  = df_players.set_index('ID')['Joueur'].to_dict()
            id_to_price = df_players.set_index('ID')['Prix'].to_dict()

            df_tr = pd.DataFrame(transfers)
            df_tr['Acheté']     = df_tr['element_in'].map(id_to_name).fillna('Inconnu')
            df_tr['Vendu']      = df_tr['element_out'].map(id_to_name).fillna('Inconnu')
            df_tr['Prix_Achat'] = df_tr['element_in_cost'] / 10
            df_tr['Prix_Vente'] = df_tr['element_out_cost'] / 10
            df_tr['P&L (M£)']   = (df_tr['Prix_Vente'] - df_tr['Prix_Achat']).round(1)

            df_tr_display = df_tr[['event', 'Acheté', 'Prix_Achat', 'Vendu', 'Prix_Vente', 'P&L (M£)']].rename(
                columns={'event': 'GW'}
            ).sort_values('GW', ascending=False)

            def color_pl(val):
                if val > 0: return 'color: green'
                if val < 0: return 'color: red'
                return ''

            try:
                st.dataframe(
                    df_tr_display.style.map(color_pl, subset=['P&L (M£)']),
                    hide_index=True, use_container_width=True
                )
            except Exception:
                st.dataframe(df_tr_display, hide_index=True, use_container_width=True)

            t1, t2, t3 = st.columns(3)
            t1.metric("Total transferts", len(df_tr))
            total_cost = df_tr['element_transfers_cost'].sum() if 'element_transfers_cost' in df_tr.columns else 0
            t2.metric("Points de pénalité", f"-{total_cost} pts")
            best_buy_id = df_tr.loc[df_tr['Prix_Achat'].idxmin(), 'element_in'] if not df_tr.empty else None
            best_name   = id_to_name.get(best_buy_id, '?') if best_buy_id else '?'
            t3.metric("Achat le moins cher", best_name)

        else:
            st.info("Aucun transfert enregistré pour cette saison.")


# ============================================================
# --- 5. PAGE : PRÉDICTIONS XP (ML) ---
# ============================================================

elif page == "🔮 Prédictions xP":

    st.title("🔮 Prédictions xP — Machine Learning")
    st.caption(
        "RandomForestRegressor entraîné sur les stats de la saison en cours · "
        "Features : form, minutes, total_points, prix, ICT index, position, FDR · "
        "Target : ep_next (expected points FPL)"
    )

    with st.spinner("Entraînement du modèle RandomForest en cours..."):
        df_pred, pred_error = load_predictions()

    if pred_error:
        st.error(f"❌ {pred_error}")
        st.stop()

    if df_pred is None or df_pred.empty:
        st.warning("Aucune donnée disponible pour les prédictions.")
        st.stop()

    # ── Appliquer filtres sidebar ──
    df_filtered = df_pred.copy()
    if pred_pos_filter != "Tous":
        df_filtered = df_filtered[df_filtered['Position'] == pred_pos_filter]
    if pred_search.strip():
        df_filtered = df_filtered[
            df_filtered['Joueur'].str.contains(pred_search.strip(), case=False, na=False)
        ]

    # ── Top 3 Recommandations ──
    st.subheader("🏆 Top 3 Recommandations (toutes positions)")
    top3 = df_pred.nlargest(3, 'xP_Total').reset_index(drop=True)
    col_m1, col_m2, col_m3 = st.columns(3)
    for i, col_m in enumerate([col_m1, col_m2, col_m3]):
        if i < len(top3):
            r = top3.iloc[i]
            col_m.metric(
                label=f"#{i + 1} · {r['Joueur']} ({r['Position']})",
                value=f"{r['xP_Total']} pts",
                delta=f"{r['Equipe']} · {r['Prix']}M£",
            )

    st.markdown("---")

    # ── Scatter Plot : Prix vs xP_Total ──
    st.subheader("📈 Prix vs xP Total")
    if not df_filtered.empty:
        # Taille minimum pour que les 0 restent visibles
        df_filtered = df_filtered.copy()
        df_filtered['_size'] = (df_filtered['xP_Total'] + 0.5).clip(lower=0.5)

        fig_pred = px.scatter(
            df_filtered,
            x='Prix',
            y='xP_Total',
            color='Position',
            size='_size',
            hover_name='Joueur',
            hover_data={'Equipe': True, 'xP_GW1': True, 'xP_GW2': True, '_size': False},
            labels={'Prix': 'Prix (M£)', 'xP_Total': 'xP Total (GW+1 + GW+2)'},
            color_discrete_map={
                'GKP': '#636EFA',
                'DEF': '#00CC96',
                'MID': '#EF553B',
                'FWD': '#FFA15A',
            },
        )
        fig_pred.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='white')))
        fig_pred.update_layout(height=500, legend_title_text='Position')
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Aucun joueur ne correspond aux filtres sélectionnés.")

    st.markdown("---")

    # ── Tableau classement ──
    st.subheader(f"📋 Classement — {len(df_filtered)} joueurs affichés")
    if not df_filtered.empty:
        df_table = (
            df_filtered[['Joueur', 'Equipe', 'Position', 'Prix', 'xP_GW1', 'xP_GW2', 'xP_Total']]
            .sort_values('xP_Total', ascending=False)
            .rename(columns={'xP_GW1': 'xP GW+1', 'xP_GW2': 'xP GW+2'})
            .reset_index(drop=True)
        )
        df_table.index = df_table.index + 1
        st.dataframe(df_table, use_container_width=True, height=500)


# ============================================================
# --- 6. PAGE : PRÉDICTIONS V6 (modèles pkl) ---
# ============================================================

elif page == "🧪 Prédictions V6":

    st.title("🧪 Prédictions V6 — Modèles ML pré-entraînés")
    st.caption(
        "Modèles pkl V6 par position (GK/DEF/MID/FWD) · "
        "Features SHAP · xP_GW2 = xP_GW1 × 0.95"
    )

    with st.spinner("🔄 Chargement des prédictions V6..."):
        df_v6, v6_err = get_xp_predictions()

    if v6_err:
        st.warning(f"⚠️ {v6_err}")
        st.stop()

    if df_v6 is None or df_v6.empty:
        st.warning("Aucune prédiction V6 disponible.")
        st.stop()

    # Infos modèles disponibles
    loaded_models = [p for p in ['GK', 'DEF', 'MID', 'FWD'] if os.path.exists(f'model_v6_{p}.pkl')]
    if loaded_models:
        st.success(f"Modèles chargés : {', '.join(loaded_models)}")
    missing = [p for p in ['GK', 'DEF', 'MID', 'FWD'] if not os.path.exists(f'model_v6_{p}.pkl')]
    if missing:
        st.warning(f"Modèles manquants : {', '.join(missing)} — xP = 0 pour ces positions.")

    # ── BLOC A : Top 3 Recommandations ──
    st.subheader("🏆 Top 3 Recommandations")
    top3_v6 = df_v6.nlargest(3, 'xP_Total').reset_index(drop=True)
    pos_emoji = {'GK': '🧤', 'DEF': '🛡️', 'MID': '🎯', 'FWD': '⚽'}
    col1, col2, col3 = st.columns(3)
    for i, col_m in enumerate([col1, col2, col3]):
        if i < len(top3_v6):
            r = top3_v6.iloc[i]
            emj = pos_emoji.get(r['position'], '⚽')
            dgw_tag = " 🔥DGW" if r['is_double_gw'] else ""
            col_m.metric(
                label=f"{emj} {r['name']}{dgw_tag} ({r['team']})",
                value=f"xP: {r['xP_Total']:.1f}",
                delta=f"£{r['price']}m — {r['position']}"
            )

    st.markdown("---")

    # ── BLOC B : Filtres ──
    col_pos_v6, col_prix_v6, col_search_v6 = st.columns(3)
    with col_pos_v6:
        pos_filter_v6 = st.selectbox(
            "Position", ["Tous", "GK", "DEF", "MID", "FWD"],
            index=["Tous", "GK", "DEF", "MID", "FWD"].index(v6_pos_sidebar)
            if v6_pos_sidebar in ["Tous", "GK", "DEF", "MID", "FWD"] else 0,
            key="main_v6_pos"
        )
    with col_prix_v6:
        max_price_v6 = st.slider(
            "Prix max (£m)", 4.0, 16.0, v6_price_sidebar, step=0.5, key="main_v6_prix"
        )
    with col_search_v6:
        search_v6 = st.text_input(
            "Rechercher joueur", value=v6_search_sidebar,
            placeholder="ex: Salah", key="main_v6_search"
        )

    df_v6_filt = df_v6.copy()
    if pos_filter_v6 != "Tous":
        df_v6_filt = df_v6_filt[df_v6_filt['position'] == pos_filter_v6]
    df_v6_filt = df_v6_filt[df_v6_filt['price'] <= max_price_v6]
    if search_v6.strip():
        df_v6_filt = df_v6_filt[
            df_v6_filt['name'].str.contains(search_v6.strip(), case=False, na=False)
        ]

    # ── BLOC C : Scatter Prix vs xP ──
    st.markdown("---")
    st.subheader("📈 Prix vs xP Total — 2 prochaines GW")
    if not df_v6_filt.empty:
        df_plot_v6 = df_v6_filt.copy()
        df_plot_v6['_sz'] = (df_plot_v6['xP_Total'] + 0.5).clip(lower=0.5)
        fig_v6 = px.scatter(
            df_plot_v6,
            x='price', y='xP_Total',
            color='position',
            size='_sz',
            hover_name='name',
            hover_data={
                'team': True, 'xP_GW1': True, 'xP_GW2': True,
                'chance_of_playing': True, '_sz': False, 'price': True
            },
            labels={'price': 'Prix (£m)', 'xP_Total': 'xP Total'},
            title="Prix vs xP Total — 2 prochaines GW",
            color_discrete_map={
                'GK': '#636EFA', 'DEF': '#00CC96', 'MID': '#EF553B', 'FWD': '#FFA15A'
            },
        )
        fig_v6.add_vline(
            x=10.0, line_dash="dash", line_color="gray",
            annotation_text="Budget moy. £10m", annotation_position="top right"
        )
        fig_v6.update_traces(marker=dict(opacity=0.8))
        fig_v6.update_layout(height=500, legend_title_text='Position')
        st.plotly_chart(fig_v6, use_container_width=True)
    else:
        st.info("Aucun joueur ne correspond aux filtres sélectionnés.")

    # ── BLOC D : Tableau complet ──
    st.markdown("---")
    st.subheader(f"📋 Classement — {len(df_v6_filt)} joueurs")
    if not df_v6_filt.empty:
        def _fmt_dispo(ch):
            if ch is None or ch >= 100:
                return "✅ 100%"
            elif ch >= 75:
                return f"⚠️ {ch}%"
            elif ch > 0:
                return f"⚠️ {ch}%"
            else:
                return "❌ 0%"

        df_show_v6 = df_v6_filt[
            ['name', 'team', 'position', 'price', 'xP_GW1', 'xP_GW2', 'xP_Total',
             'chance_of_playing', 'is_double_gw']
        ].copy()
        df_show_v6['Joueur'] = df_show_v6.apply(
            lambda r: f"{r['name']} 🔥" if r['is_double_gw'] else r['name'], axis=1
        )
        df_show_v6['Prix'] = df_show_v6['price'].apply(lambda x: f"£{x:.1f}m")
        df_show_v6['Dispo'] = df_show_v6['chance_of_playing'].apply(_fmt_dispo)
        df_show_v6['xP GW+1'] = df_show_v6['xP_GW1'].round(1)
        df_show_v6['xP GW+2'] = df_show_v6['xP_GW2'].round(1)
        df_show_v6['Total'] = df_show_v6['xP_Total'].round(1)

        df_table_v6 = (
            df_show_v6[['Joueur', 'team', 'position', 'Prix', 'xP GW+1', 'xP GW+2', 'Total', 'Dispo']]
            .rename(columns={'team': 'Équipe', 'position': 'Pos'})
            .sort_values('Total', ascending=False)
            .reset_index(drop=True)
        )
        df_table_v6.index += 1
        st.dataframe(df_table_v6, use_container_width=True, height=500)


# ============================================================
# --- 7. PAGE : TRANSFERTS SUGGÉRÉS ---
# ============================================================

elif page == "🔄 Transferts":

    st.title("🔄 Transferts Suggérés")
    st.caption("Basé sur les prédictions V6 · Identifie les 3 joueurs les moins performants et suggère des remplaçants.")

    # ── BLOC A : Team ID ──
    team_id_tr_raw = st.number_input(
        "Ton Team ID FPL", min_value=1, max_value=99999999,
        value=int(st.session_state.get('tr_last_id', 1)),
        step=1, format="%d",
        help="Entre 1 pour accès owner direct"
    )
    team_id_tr = OWNER_ID if int(team_id_tr_raw) == 1 else int(team_id_tr_raw)
    if team_id_tr == OWNER_ID:
        st.info(f"👑 Owner connecté — Team ID : {OWNER_ID}")

    if st.button("🔍 Analyser mon équipe et suggérer des transferts", key="btn_tr_analyze"):
        st.session_state['tr_last_id'] = int(team_id_tr_raw)
        st.session_state['tr_analyzed_id'] = team_id_tr

    analyzed_id = st.session_state.get('tr_analyzed_id', None)

    if not analyzed_id:
        st.info("Entre ton Team ID FPL et clique sur **Analyser mon équipe**.")
        st.stop()

    # Chargement données
    with st.spinner("🔄 Chargement des prédictions V6..."):
        df_v6_tr, v6_err_tr = get_xp_predictions()

    if v6_err_tr:
        st.error(f"❌ {v6_err_tr}")
        st.stop()

    with st.spinner(f"Chargement de la team {analyzed_id}..."):
        info_tr, _, picks_tr, _, _ = load_my_team(analyzed_id)

    if info_tr is None or picks_tr is None:
        st.error("❌ Team introuvable. Vérifie ton Team ID.")
        st.stop()

    picks = picks_tr.get('picks', [])
    entry_hist_tr = picks_tr.get('entry_history', {})
    bank_tr = entry_hist_tr.get('bank', 0) / 10

    # Enrichissement du squad avec xP V6
    v6_by_id = (
        df_v6_tr.set_index('id')
        if (df_v6_tr is not None and not df_v6_tr.empty)
        else pd.DataFrame()
    )

    squad_rows_tr = []
    for pick in picks:
        pid = pick['element']
        xp_t = 0.0; price_p = 0.0; pos_p = 'MID'
        name_p = str(pid); team_p = ''
        if not v6_by_id.empty and pid in v6_by_id.index:
            rv = v6_by_id.loc[pid]
            xp_t = float(rv['xP_Total']); price_p = float(rv['price'])
            pos_p = str(rv['position']); name_p = str(rv['name'])
            team_p = str(rv['team'])
        elif not df_players.empty and pid in df_players['ID'].values:
            rp = df_players[df_players['ID'] == pid].iloc[0]
            price_p = float(rp['Prix'])
            pos_p = str(rp['Position']).replace('GKP', 'GK')
            name_p = str(rp['Joueur']); team_p = str(rp['Equipe'])
        squad_rows_tr.append({
            'id': pid, 'name': name_p, 'team': team_p,
            'position': pos_p, 'price': price_p, 'xP_Total': xp_t,
        })

    df_squad_tr = pd.DataFrame(squad_rows_tr)
    squad_ids_tr = set(df_squad_tr['id'].tolist())

    # ── BLOC B : Affichage squad ──
    st.markdown("---")
    st.subheader(f"📋 {info_tr.get('name', 'Mon Équipe')}")

    b1_tr, b2_tr, b3_tr = st.columns(3)
    b1_tr.info(f"💰 En banque : **£{bank_tr:.1f}m**")
    b2_tr.info(f"📊 xP moyen squad : **{df_squad_tr['xP_Total'].mean():.1f}**")
    b3_tr.info(f"👥 Joueurs chargés : **{len(df_squad_tr)}**")

    df_squad_disp = df_squad_tr[['name', 'team', 'position', 'price', 'xP_Total']].copy()
    df_squad_disp.columns = ['Joueur', 'Équipe', 'Pos', 'Prix (£m)', 'xP Total']
    df_squad_disp['xP Total'] = df_squad_disp['xP Total'].round(2)
    df_squad_disp = df_squad_disp.sort_values('xP Total', ascending=False).reset_index(drop=True)
    df_squad_disp.index += 1
    st.dataframe(df_squad_disp, hide_index=False, use_container_width=True)

    if df_v6_tr is None or df_v6_tr.empty:
        st.warning("⚠️ Prédictions V6 indisponibles — suggestions de transferts impossibles.")
        st.stop()

    # ── BLOC C : Suggestions de transferts ──
    st.markdown("---")
    st.subheader("💡 3 Transferts Suggérés")

    candidates_out = df_squad_tr.nsmallest(3, 'xP_Total').reset_index(drop=True)
    total_gain_tr = 0.0
    current_bank_tr = bank_tr

    for _, rout in candidates_out.iterrows():
        pos_need = rout['position']
        price_out = float(rout['price'])
        budget_avail = price_out + current_bank_tr

        # Chercher meilleur joueur dispo, même position, dans le budget, pas dans squad
        mask_tr = (
            (df_v6_tr['position'] == pos_need) &
            (df_v6_tr['price'] <= budget_avail + 0.05) &
            (~df_v6_tr['id'].isin(squad_ids_tr)) &
            (df_v6_tr['chance_of_playing'] >= 75)
        )
        best_in = df_v6_tr[mask_tr].nlargest(1, 'xP_Total')

        # Fallback sans contrainte budget/dispo
        if best_in.empty:
            mask_fb = (
                (df_v6_tr['position'] == pos_need) &
                (~df_v6_tr['id'].isin(squad_ids_tr))
            )
            best_in = df_v6_tr[mask_fb].nlargest(1, 'xP_Total')

        if best_in.empty:
            st.write(f"Aucun remplaçant trouvé pour {rout['name']} ({pos_need}).")
            continue

        rin = best_in.iloc[0]
        gain_tr = float(rin['xP_Total']) - float(rout['xP_Total'])
        total_gain_tr += gain_tr
        price_diff_tr = float(rin['price']) - price_out
        current_bank_tr -= price_diff_tr

        col_out_tr, col_arr_tr, col_in_tr = st.columns([5, 1, 5])
        with col_out_tr:
            st.markdown(f"#### ❌ {rout['name']}")
            st.markdown(
                f"xP actuel : **{rout['xP_Total']:.1f}** &nbsp;|&nbsp; "
                f"Prix : **£{rout['price']:.1f}m** &nbsp;|&nbsp; {rout['team']}",
                unsafe_allow_html=True
            )
        with col_arr_tr:
            st.markdown(
                "<p style='font-size:2em;text-align:center;margin-top:12px'>→</p>",
                unsafe_allow_html=True
            )
        with col_in_tr:
            st.markdown(f"#### ✅ {rin['name']}")
            gain_str = f"+{gain_tr:.1f}" if gain_tr >= 0 else f"{gain_tr:.1f}"
            st.markdown(
                f"xP prédit : **{rin['xP_Total']:.1f}** ({gain_str}) &nbsp;|&nbsp; "
                f"Prix : **£{rin['price']:.1f}m** &nbsp;|&nbsp; {rin['team']}",
                unsafe_allow_html=True
            )
        st.markdown("---")

    m1_tr, m2_tr = st.columns(2)
    m1_tr.metric("Gain xP estimé", f"+{total_gain_tr:.1f} pts/GW")
    m2_tr.metric("Budget après transferts", f"£{current_bank_tr:.1f}m")


# ============================================================
# --- 8. PAGE : CAPITAINE SUGGÉRÉ ---
# ============================================================

elif page == "⚽ Capitaine":

    st.title("⚽ Capitaine Suggéré")
    st.caption(
        "Score = xP_GW1 × 2 × elo_win_prob × chance_of_playing × (1 / FDR_next)"
    )

    with st.spinner("🔄 Chargement des prédictions V6..."):
        df_v6_cap, v6_err_cap = get_xp_predictions()

    if v6_err_cap:
        st.error(f"❌ {v6_err_cap}")
        st.stop()

    if df_v6_cap is None or df_v6_cap.empty:
        st.warning("Aucune prédiction disponible.")
        st.stop()

    # ── BLOC A : Calcul score capitaine ──
    df_cap = df_v6_cap.copy()
    df_cap['captain_score'] = (
        df_cap['xP_GW1'] * 2.0
        * df_cap['elo_win_prob'].clip(0.05, 0.95)
        * (df_cap['chance_of_playing'] / 100.0).clip(0.01, 1.0)
        * (1.0 / df_cap['fdr_next'].clip(1, 5).astype(float))
    ).round(3)

    df_cap = df_cap.sort_values('captain_score', ascending=False).reset_index(drop=True)

    top_cap = df_cap.iloc[0] if len(df_cap) > 0 else None
    vice_cap = df_cap.iloc[1] if len(df_cap) > 1 else None

    # ── BLOC B : Recommandation principale ──
    if top_cap is not None:
        dgw_tag = " 🔥 Double GW" if top_cap['is_double_gw'] else ""
        st.success(
            f"**🥇 CAPITAINE RECOMMANDÉ : {top_cap['name']}{dgw_tag}**\n\n"
            f"Score capitaine : **{top_cap['captain_score']:.2f}**  \n"
            f"xP × 2 : **{top_cap['xP_GW1'] * 2:.1f} pts**  \n"
            f"Prochain match : **{top_cap['opp_name']}** — FDR : {top_cap['fdr_next']}  \n"
            f"Forme : {top_cap['form']} | Dispo : {top_cap['chance_of_playing']}%  \n"
            f"Probabilité victoire Elo : {top_cap['elo_win_prob']:.0%}"
        )

    # ── BLOC C : Top 5 tableau ──
    st.markdown("---")
    st.subheader("🏆 Top 5 Candidats Capitaine")

    top5_cap = df_cap.head(5).copy()
    top5_cap.insert(0, '#', range(1, len(top5_cap) + 1))
    top5_cap['xP×2'] = (top5_cap['xP_GW1'] * 2).round(1)
    top5_cap['Elo win%'] = top5_cap['elo_win_prob'].apply(lambda x: f"{x:.0%}")
    top5_cap['Score'] = top5_cap['captain_score'].round(2)
    top5_cap['Dispo'] = top5_cap['chance_of_playing'].apply(lambda x: f"{x}%")

    cap_table = top5_cap[[
        '#', 'name', 'xP×2', 'opp_name', 'fdr_next',
        'Elo win%', 'Dispo', 'Score'
    ]].rename(columns={
        'name': 'Joueur', 'opp_name': 'Adversaire', 'fdr_next': 'FDR'
    })
    st.dataframe(cap_table, hide_index=True, use_container_width=True)

    # ── BLOC D : Capitaine vs Vice-capitaine ──
    if top_cap is not None and vice_cap is not None:
        st.markdown("---")
        st.subheader("⚖️ Capitaine vs Vice-capitaine")

        col_cap_l, col_cap_mid, col_cap_r = st.columns([4, 1, 4])

        with col_cap_l:
            st.markdown(f"### 🥇 {top_cap['name']}")
            st.write(f"**Équipe :** {top_cap['team']}")
            st.write(f"**xP GW+1 :** {top_cap['xP_GW1']:.1f} → **{top_cap['xP_GW1'] * 2:.1f}** (×2 capitaine)")
            st.write(f"**Adversaire :** {top_cap['opp_name']}")
            st.write(f"**FDR :** {top_cap['fdr_next']} | **Forme :** {top_cap['form']}")
            st.write(f"**Dispo :** {top_cap['chance_of_playing']}%")
            st.metric("Score capitaine", f"{top_cap['captain_score']:.3f}")

        with col_cap_mid:
            diff_score = top_cap['captain_score'] - vice_cap['captain_score']
            st.markdown(
                f"<div style='text-align:center;margin-top:40px'>"
                f"<b>Δ</b><br>+{diff_score:.3f}</div>",
                unsafe_allow_html=True
            )

        with col_cap_r:
            st.markdown(f"### 🥈 {vice_cap['name']}")
            st.write(f"**Équipe :** {vice_cap['team']}")
            st.write(f"**xP GW+1 :** {vice_cap['xP_GW1']:.1f} → **{vice_cap['xP_GW1'] * 2:.1f}** (×2 capitaine)")
            st.write(f"**Adversaire :** {vice_cap['opp_name']}")
            st.write(f"**FDR :** {vice_cap['fdr_next']} | **Forme :** {vice_cap['form']}")
            st.write(f"**Dispo :** {vice_cap['chance_of_playing']}%")
            st.metric("Score capitaine", f"{vice_cap['captain_score']:.3f}")
