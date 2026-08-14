"""
sentiment_test.py -- Test de faisabilite : sentiment news + joueurs FPL

OBJECTIF : avant de construire un pipeline complet (scraping continu +
NLP + integration aux predictions xP), verifier si la couverture presse
est suffisante pour que ce soit rentable.

Etape 1 : scrape le flux RSS BBC Sport Football, filtre les articles qui
          mentionnent un joueur Premier League connu (liste FPL live).
Etape 2 : sentiment (distilbert-base-uncased-finetuned-sst-2-english) +
          mots-cles blessure/forme sur les articles filtres.
Etape 3 : rapport de faisabilite + verdict (<30 joueurs = marginal,
          >100 joueurs = ca vaut le coup).

Usage : python sentiment_test.py
"""
import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')


def pip_install(*pkgs):
    for p in pkgs:
        mod = p.split('[')[0].replace('-', '_')
        try:
            __import__(mod)
        except ImportError:
            print(f"   Installation de {p} ...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', p, '-q'])


pip_install('feedparser', 'requests')

import feedparser
import requests

RSS_URL = "https://feeds.bbci.co.uk/sport/football/rss.xml"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
N_ARTICLES = 30

INJURY_KEYWORDS = ['injury', 'injured', 'doubt', 'knock', 'fitness', 'sidelined']
POSITIVE_KEYWORDS = ['fit', 'available', 'return', 'starts', 'confirmed']


# =============================================================================
# Etape 1 -- Scraping RSS + filtrage joueurs FPL
# =============================================================================
def fetch_articles(n=N_ARTICLES):
    print(f"[1/3] Scraping RSS BBC Sport Football ...")
    feed = feedparser.parse(RSS_URL)
    if getattr(feed, 'bozo', False) and not feed.entries:
        raise RuntimeError(f"Flux RSS illisible : {feed.get('bozo_exception')}")

    articles = []
    for e in feed.entries[:n]:
        articles.append({
            'title': e.get('title', ''),
            'summary': re.sub('<[^<]+?>', '', e.get('summary', '')),  # strip HTML
            'published': e.get('published', e.get('updated', '')),
            'link': e.get('link', ''),
        })
    print(f"   {len(articles)} articles recuperes.")
    return articles


def fetch_player_names():
    print("   Chargement de la liste des joueurs FPL (bootstrap-static) ...")
    resp = requests.get(BOOTSTRAP_URL, timeout=15,
                         headers={'User-Agent': 'Mozilla/5.0 (FPL-Sentiment-Test)'})
    resp.raise_for_status()
    elements = resp.json()['elements']

    players = []
    for p in elements:
        web_name = p.get('web_name', '').strip()
        second_name = p.get('second_name', '').strip()
        full_name = f"{p.get('first_name', '')} {second_name}".strip()
        # candidats de matching : web_name (surnom FPL, le plus utilise en presse),
        # nom de famille, nom complet -- on ignore les candidats trop courts
        # (<=3 caracteres) pour eviter les faux positifs (ex: "Cox", "Ait").
        candidates = {c for c in {web_name, second_name, full_name} if len(c) > 3}
        if not candidates:
            continue
        players.append({
            'id': p['id'], 'web_name': web_name, 'full_name': full_name,
            'candidates': candidates,
        })
    print(f"   {len(players)}/{len(elements)} joueurs FPL charges "
          f"(candidats de matching valides).")
    return players, len(elements)


def build_matcher(players):
    """Regex word-boundary (insensible a la casse) par candidat -> id joueur."""
    patterns = []
    for pl in players:
        for cand in pl['candidates']:
            pat = re.compile(r'\b' + re.escape(cand) + r'\b', re.IGNORECASE)
            patterns.append((pat, pl))
    return patterns


def match_articles_to_players(articles, patterns):
    print("[2/3] Filtrage : articles mentionnant un joueur FPL connu ...")
    matches = []
    for art in articles:
        text = f"{art['title']} {art['summary']}"
        found_players = []
        for pat, pl in patterns:
            if pat.search(text):
                found_players.append(pl)
        if found_players:
            # dedupe (un joueur peut matcher via plusieurs candidats)
            seen_ids, uniq = set(), []
            for pl in found_players:
                if pl['id'] not in seen_ids:
                    seen_ids.add(pl['id'])
                    uniq.append(pl)
            matches.append({**art, 'players': uniq})
    n_player_mentions = sum(len(m['players']) for m in matches)
    print(f"   {len(matches)}/{len(articles)} articles matches "
          f"({n_player_mentions} mentions joueur au total).")
    return matches


# =============================================================================
# Etape 2 -- Sentiment + mots-cles
# =============================================================================
def load_sentiment_pipeline():
    print("\n   Chargement du modele de sentiment "
          "(distilbert-base-uncased-finetuned-sst-2-english) ...")
    pip_install('transformers', 'torch')
    from transformers import pipeline
    return pipeline('sentiment-analysis',
                     model='distilbert-base-uncased-finetuned-sst-2-english')


def analyse_sentiment(matches, sentiment_pipe):
    print("[2/3] Sentiment + mots-cles sur les articles matches ...")
    for m in matches:
        text = f"{m['title']}. {m['summary']}"
        text_lower = text.lower()

        try:
            res = sentiment_pipe(text[:512])[0]  # troncature = limite du modele
            m['sentiment_label'] = res['label']
            m['sentiment_score'] = round(float(res['score']), 3)
        except Exception as e:
            m['sentiment_label'] = 'ERROR'
            m['sentiment_score'] = 0.0

        m['injury_keywords_found'] = [k for k in INJURY_KEYWORDS if k in text_lower]
        m['positive_keywords_found'] = [k for k in POSITIVE_KEYWORDS if k in text_lower]
    return matches


# =============================================================================
# Etape 3 -- Rapport de faisabilite
# =============================================================================
def print_report(all_articles, matches, players_total=581):
    print("\n" + "=" * 78)
    print("   RAPPORT DE FAISABILITE -- Sentiment News x Joueurs FPL")
    print("=" * 78)

    unique_players = {}
    for m in matches:
        for pl in m['players']:
            unique_players.setdefault(pl['id'], pl['web_name'])

    print(f"\n[1] Volume")
    print(f"    Articles recuperes (RSS BBC Sport Football) : {len(all_articles)}")
    print(f"    Articles matches a >=1 joueur FPL            : {len(matches)}")
    print(f"    Joueurs UNIQUES mentionnes                   : {len(unique_players)}")

    print(f"\n[2] Top 10 meilleurs matchs")
    top = sorted(matches, key=lambda m: m['sentiment_score'], reverse=True)[:10]
    if not top:
        print("    (aucun match)")
    for i, m in enumerate(top, 1):
        players_str = ', '.join(pl['web_name'] for pl in m['players'])
        kw = m['injury_keywords_found'] + m['positive_keywords_found']
        kw_str = f"[{', '.join(kw)}]" if kw else "[-]"
        print(f"    {i:2d}. {players_str:<25} | {m['sentiment_label']:<8} "
              f"({m['sentiment_score']:.2f}) {kw_str}")
        print(f"        \"{m['title']}\"")

    print(f"\n[3] Joueurs mentionnes (liste complete)")
    print(f"    {', '.join(sorted(unique_players.values()))}" if unique_players else "    (aucun)")

    # ---- Estimation hebdo -------------------------------------------------
    n_unique = len(unique_players)
    print(f"\n[4] Estimation")
    print(f"    NB : ce test couvre les {len(all_articles)} derniers titres du flux "
          f"BBC Sport Football au moment du run (pas litteralement 7 jours de flux --")
    print(f"    la frequence de publication de BBC varie ; a considerer comme un "
          f"echantillon indicatif, pas une mesure hebdo exacte).")
    print(f"    Sur {players_total} joueurs FPL au total, {n_unique} auraient une "
          f"info sentiment sur cet echantillon ({n_unique/players_total*100:.1f}%).")

    print("\n" + "=" * 78)
    print("   VERDICT")
    print("=" * 78)
    if n_unique < 30:
        print(f"   {n_unique} joueurs couverts (<30) -> GAIN MARGINAL.")
        print(f"   Couverture trop faible pour justifier un pipeline complet : "
              f"la grande majorite des 581 joueurs FPL n'auraient jamais d'info")
        print(f"   sentiment, l'integration aux predictions xP aurait un impact "
              f"quasi nul sur la majorite des decisions.")
    elif n_unique > 100:
        print(f"   {n_unique} joueurs couverts (>100) -> CA VAUT LE COUP.")
        print(f"   Couverture large : construire le pipeline complet (scraping "
              f"continu multi-sources + integration aux features xP) est justifie.")
    else:
        print(f"   {n_unique} joueurs couverts (entre 30 et 100) -> ZONE GRISE.")
        print(f"   Ni marginal ni clairement rentable -- envisager d'elargir les "
              f"sources RSS (Sky Sports, The Athletic, etc.) avant de trancher.")
    print("=" * 78)


def main():
    articles = fetch_articles()
    players, n_total_players = fetch_player_names()
    patterns = build_matcher(players)
    matches = match_articles_to_players(articles, patterns)

    if matches:
        sentiment_pipe = load_sentiment_pipeline()
        matches = analyse_sentiment(matches, sentiment_pipe)
    else:
        print("   Aucun article matche -- sentiment analysis sautee.")

    print_report(articles, matches, players_total=n_total_players)


if __name__ == '__main__':
    main()
