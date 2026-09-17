"""Scrape la liste des Top 1000 managers "Elite" de LiveFPL
(https://plan.livefpl.net/elite) et sauvegarde Rank, Name, Alltime score,
et les scores par saison dans data/elite_managers.csv.

Le tableau est entièrement présent dans le HTML statique (DataTables ne
fait qu'enrichir l'affichage côté client) : une seule requête HTTP suffit,
pas besoin de Selenium ni de pagination.
"""
import csv
import os
import re
import sys

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

URL = "https://plan.livefpl.net/elite"
OUTPUT_PATH = os.path.join("data", "elite_managers.csv")
ENTRY_ID_RE = re.compile(r"/entry/(\d+)/history")


def _build_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    })
    return session


def _clean_cell(text):
    """Nettoie une cellule : enlève les séparateurs de milliers si c'est un nombre."""
    text = text.strip()
    if text in ("", "-", "—"):
        return ""
    stripped = text.replace(",", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", stripped):
        return stripped
    return text


def fetch_elite_page(session):
    print(f"→ Téléchargement de {URL} ...")
    r = session.get(URL, timeout=20)
    r.raise_for_status()
    print(f"  OK ({len(r.content):,} octets reçus)")
    return r.text


def parse_elite_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="main")
    if table is None:
        raise ValueError("Impossible de trouver le tableau #main sur la page — "
                          "la structure du site a peut-être changé.")

    thead = table.find("thead")
    if thead is not None:
        raw_headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]
    else:
        raw_headers = [c.get_text(strip=True) for c in table.find("tr").find_all(["th", "td"])]

    # La 2e colonne (index 1) est l'icône "favori", sans intérêt.
    fav_col_idx = None
    for i, h in enumerate(raw_headers):
        if h == "" and i > 0:
            fav_col_idx = i
            break

    headers = [h if h else None for h in raw_headers]
    if fav_col_idx is not None:
        headers[fav_col_idx] = "_skip"

    tbody = table.find("tbody")
    if tbody is not None:
        rows = tbody.find_all("tr")
    else:
        # Sur ce site, les <th> de l'en-tête ne sont pas enveloppés dans un
        # <tr> : table.find_all("tr") ne renvoie donc QUE les lignes de
        # données. On ne retire une éventuelle 1ère ligne que si elle
        # contient elle-même des <th> (cas d'un tableau sans <thead> du tout).
        rows = table.find_all("tr")
        if thead is None and rows and rows[0].find("th"):
            rows = rows[1:]

    managers = []
    total = len(rows)
    if total == 0:
        raise ValueError("Aucune ligne trouvée dans le tableau — vérifie la page manuellement.")

    for i, row in enumerate(rows, start=1):
        cells = row.find_all("td")
        if len(cells) != len(headers):
            print(f"  ⚠️ Ligne {i} ignorée (nb colonnes inattendu : {len(cells)} vs {len(headers)})")
            continue

        record = {}
        entry_id = ""
        for header, cell in zip(headers, cells):
            if header in (None, "_skip"):
                continue
            link = cell.find("a", href=True)
            if link:
                m = ENTRY_ID_RE.search(link["href"])
                if m:
                    entry_id = m.group(1)
            record[header] = _clean_cell(cell.get_text(strip=True))

        record["FPL_Entry_ID"] = entry_id
        managers.append(record)

        if i % 100 == 0 or i == total:
            pct = 100 * i / total
            print(f"  Parsing... {i}/{total} managers ({pct:.0f}%)", flush=True)

    return managers


def save_to_csv(managers, path):
    if not managers:
        print("⚠️ Aucune donnée à sauvegarder.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ordre de colonnes stable, avec FPL_Entry_ID en dernier.
    fieldnames = []
    for m in managers:
        for k in m.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in managers:
            writer.writerow(m)

    print(f"✅ {len(managers)} managers sauvegardés dans {path}")


def main():
    session = _build_session()

    try:
        html = fetch_elite_page(session)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement de la page : {e}")
        sys.exit(1)

    try:
        managers = parse_elite_table(html)
    except Exception as e:
        print(f"❌ Erreur lors du parsing du tableau : {e}")
        sys.exit(1)

    save_to_csv(managers, OUTPUT_PATH)


if __name__ == "__main__":
    main()
