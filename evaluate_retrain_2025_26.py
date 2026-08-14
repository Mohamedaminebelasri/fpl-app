"""
evaluate_retrain_2025_26.py

Evalue proprement (walk-forward + holdout, SANS fuite de donnees) l'apport
du backfill 2025-26, en reproduisant exactement la methodologie de
best_model_v6.py (memes WALK_FOLDS structure, meme train_model/evaluate/
top10_accuracy), decalee d'une saison :
  - Walk-forward : 2024-25 (3 folds) + 2025-26 (2 folds)
    [equivalent de l'original : 2023-24 (3 folds) + 2024-25 (2 folds)]
  - Holdout final : train sur tout sauf 2025-26 GW>32, test sur 2025-26 GW>=33
    [equivalent de l'original : train sauf 2024-25 GW>32, test 2024-25 GW>=33]

IMPORTANT : les modeles entraines ici sont TEMPORAIRES (non sauvegardes) --
le modele de production model_v6_*.pkl a ete entraine sur les 10 saisons
COMPLETES (target compris pour la periode de test ci-dessous), donc on ne
peut pas l'evaluer honnetement sur ses propres donnees d'entrainement.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
FIXED_PARAMS = dict(n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')

WALK_FOLDS_NEW = [
    ('2024-25', 1, 10),
    ('2024-25', 11, 20),
    ('2024-25', 21, 38),
    ('2025-26', 1, 19),
    ('2025-26', 20, 38),
]


def load_optuna(pos):
    return pickle.load(open(f'optuna_XGB_{pos}.pkl', 'rb'))


def train_model(df_tr, pos, features, optuna_params, sw=1.5):
    d = df_tr[df_tr['position'] == pos]
    if len(d) < 50:
        return None
    X = d[features].fillna(0).values
    y = d['target'].values
    w = np.where(y >= 8, sw, 1.0)
    mdl = XGBRegressor(**{**optuna_params, **FIXED_PARAMS})
    mdl.fit(X, y, sample_weight=w, verbose=False)
    return mdl


def predict_all(models, df_te, features):
    preds = pd.Series(np.nan, index=df_te.index)
    for pos, mdl in models.items():
        if mdl is None:
            continue
        mask = df_te['position'] == pos
        if mask.sum() == 0:
            continue
        preds.loc[mask] = mdl.predict(df_te.loc[mask, features].fillna(0).values)
    return preds


def evaluate(yt, yp):
    return (mean_absolute_error(yt, yp),
            mean_squared_error(yt, yp) ** 0.5,
            r2_score(yt, yp))


def top10_accuracy(df_ev):
    hits, n_gw = 0, 0
    for gw in sorted(df_ev['GW'].unique()):
        g = df_ev[df_ev['GW'] == gw]
        if len(g) < 10:
            continue
        hits += len(set(g.nlargest(10, 'y_pred').index) &
                    set(g.nlargest(10, 'target').index))
        n_gw += 1
    return (hits / (n_gw * 10)) * 100 if n_gw > 0 else 0.0


def main():
    print("Chargement cache_v6_features.pkl (10 saisons) ...")
    df_v6 = pickle.load(open('cache_v6_features.pkl', 'rb'))
    df_v6 = df_v6.dropna(subset=['target']).copy()
    top25_features = pickle.load(open('shap_v6_top_features.pkl', 'rb'))
    print(f"   {len(df_v6):,} lignes, saisons: {sorted(df_v6['season'].unique())}")

    # ---- Walk-forward -------------------------------------------------
    print("\n[Walk-Forward] 2024-25 (3 folds) + 2025-26 (2 folds) ...")
    wf_results = []
    for fold_i, (test_season, gw_start, gw_end) in enumerate(WALK_FOLDS_NEW, 1):
        mask_tr = (df_v6['season'] != test_season) | (df_v6['GW'] < gw_start)
        mask_te = ((df_v6['season'] == test_season) &
                   (df_v6['GW'] >= gw_start) & (df_v6['GW'] <= gw_end))
        df_tr, df_te = df_v6[mask_tr].copy(), df_v6[mask_te].copy()
        if len(df_te) < 50:
            continue
        fold_models = {pos: train_model(df_tr, pos, top25_features, load_optuna(pos))
                       for pos in POSITIONS}
        df_te['y_pred'] = predict_all(fold_models, df_te, top25_features)
        df_te_c = df_te.dropna(subset=['y_pred'])
        if len(df_te_c) == 0:
            continue
        mae, rmse, r2 = evaluate(df_te_c['target'], df_te_c['y_pred'])
        t10 = top10_accuracy(df_te_c.reset_index(drop=True))
        wf_results.append({'fold': fold_i, 'season': test_season, 'gw': f'{gw_start}-{gw_end}',
                            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Top10': t10, 'n': len(df_te_c)})
        print(f"   Fold {fold_i} ({test_season} GW{gw_start}-{gw_end}): "
              f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Top-10={t10:.1f}%  (n={len(df_te_c):,})")

    wf_df = pd.DataFrame(wf_results)
    wf_mae, wf_rmse, wf_r2, wf_top10 = (wf_df['MAE'].mean(), wf_df['RMSE'].mean(),
                                         wf_df['R2'].mean(), wf_df['Top10'].mean())
    print(f"\n   -- Walk-Forward Moyen (V6 + 2025-26) --")
    print(f"   MAE={wf_mae:.4f}  RMSE={wf_rmse:.4f}  R2={wf_r2:.4f}  Top-10={wf_top10:.1f}%")

    # ---- Holdout final : train sauf 2025-26 GW>32, test 2025-26 GW>=33 --
    print("\n[Holdout final] train sauf 2025-26 GW>32, test 2025-26 GW>=33 ...")
    mask_train_final = (df_v6['season'] != '2025-26') | (df_v6['GW'] <= 32)
    mask_test_final = (df_v6['season'] == '2025-26') & (df_v6['GW'] >= 33)
    df_train_final = df_v6[mask_train_final].copy()
    df_test_final = df_v6[mask_test_final].copy()
    print(f"   Train: {len(df_train_final):,} | Test GW33+: {len(df_test_final):,}")

    final_models = {pos: train_model(df_train_final, pos, top25_features, load_optuna(pos))
                     for pos in POSITIONS}
    df_test_final['y_pred'] = predict_all(final_models, df_test_final, top25_features)
    df_ev = df_test_final.dropna(subset=['y_pred'])
    final_mae, final_rmse, final_r2 = evaluate(df_ev['target'], df_ev['y_pred'])
    final_top10 = top10_accuracy(df_ev.reset_index(drop=True))
    print(f"\n   -- Score Final (holdout 2025-26 GW33+) --")
    print(f"   MAE={final_mae:.4f}  RMSE={final_rmse:.4f}  R2={final_r2:.4f}  Top-10={final_top10:.1f}%")

    # ---- Comparaison vs baseline V6 (9 saisons, holdout 2024-25 GW33+) --
    baseline = {'MAE': 1.02, 'RMSE': None, 'R2': 0.35, 'Top10': 12.6}
    print("\n" + "=" * 78)
    print(f"{'Version':<38} {'MAE':>7} {'RMSE':>7} {'R2':>6} {'Top-10':>8}")
    print("-" * 78)
    print(f"{'V6 baseline (9 saisons, holdout 24-25)':<38} "
          f"{baseline['MAE']:>7.2f} {'  n/a':>7} {baseline['R2']:>6.2f} {baseline['Top10']:>7.1f}%")
    print(f"{'V6+2025-26 WF moyen (24-25+25-26)':<38} "
          f"{wf_mae:>7.4f} {wf_rmse:>7.4f} {wf_r2:>6.4f} {wf_top10:>7.1f}%")
    print(f"{'V6+2025-26 holdout (25-26 GW33+)':<38} "
          f"{final_mae:>7.4f} {final_rmse:>7.4f} {final_r2:>6.4f} {final_top10:>7.1f}%")
    print("=" * 78)

    def pct_gain(new, old, higher_better=False):
        if old in (None, 0):
            return "n/a"
        d = (new - old) / abs(old) * 100
        if higher_better:
            return f"{'+' if d>=0 else ''}{d:.1f}%"
        return f"{'+' if d>=0 else ''}{d:.1f}%"

    print(f"\nvs baseline V6 (holdout comparable) :")
    print(f"   MAE  : {pct_gain(final_mae, baseline['MAE'])} (negatif = mieux)")
    print(f"   R2   : {pct_gain(final_r2, baseline['R2'], higher_better=True)} (positif = mieux)")
    print(f"   Top10: {pct_gain(final_top10, baseline['Top10'], higher_better=True)} (positif = mieux)")


if __name__ == '__main__':
    main()
