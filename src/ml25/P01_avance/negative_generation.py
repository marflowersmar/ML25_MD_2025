# negative_generation.py
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Portabilidad: igual que data_processing.py
CURRENT_FILE = Path(__file__).resolve()
DEFAULT_DATA_DIR = (CURRENT_FILE.parent.parent / "datasets" / "customer_purchases").resolve()
DATA_DIR = Path(os.getenv("CUSTOMER_PURCHASES_DIR", str(DEFAULT_DATA_DIR))).resolve()

def read_csv(filename: str) -> pd.DataFrame:
    file = DATA_DIR / f"{filename}.csv"
    return pd.read_csv(file)

def _safe_mode(s: pd.Series, fallback):
    try:
        m = s.mode()
        return m.iloc[0] if not m.empty else fallback
    except Exception:
        return fallback

def _coerce_dt(s: pd.Series):
    return pd.to_datetime(s, errors="coerce")

def gen_realistic_negatives(df: pd.DataFrame, n_per_positive: int = 1, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    df = df.copy()
    for c in ["customer_signup_date", "item_release_date", "purchase_timestamp", "customer_date_of_birth"]:
        if c in df.columns:
            df[c] = _coerce_dt(df[c])

    # Tabla de items para enriquecer
    item_cols = [
        "item_id","item_title","item_category","item_price","item_img_filename",
        "item_avg_rating","item_num_ratings","item_release_date"
    ]
    item_cols = [c for c in item_cols if c in df.columns]
    items_tbl = df[item_cols].drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    # Estadísticos por cliente
    grp = df.groupby("customer_id", dropna=False)
    cust_last_ts    = grp["purchase_timestamp"].max().rename("purchase_timestamp")
    cust_med_views  = grp["customer_item_views"].median().rename("customer_item_views") if "customer_item_views" in df.columns else pd.Series(dtype="float64", name="customer_item_views")
    cust_mode_dev   = (grp["purchase_device"].apply(lambda s: _safe_mode(s, "unknown_device")).rename("purchase_device")
                       if "purchase_device" in df.columns else pd.Series(dtype="object", name="purchase_device"))
    cust_fav_cat    = grp["item_category"].apply(lambda s: _safe_mode(s, None)).rename("fav_cat")
    cust_mean_price = grp["item_price"].mean().rename("mean_price")
    cust_std_price  = grp["item_price"].std().fillna(0).rename("std_price")

    cust_stats = pd.concat(
        [cust_last_ts, cust_med_views, cust_mode_dev, cust_fav_cat, cust_mean_price, cust_std_price],
        axis=1
    ).reset_index()  # columnas: customer_id, purchase_timestamp, customer_item_views, purchase_device, fav_cat, mean_price, std_price

    # Globales de respaldo
    global_med_views = float(df["customer_item_views"].median()) if "customer_item_views" in df.columns else 0.0
    global_device = _safe_mode(df["purchase_device"], "unknown_device") if "purchase_device" in df.columns else "unknown_device"

    # Items ya comprados por cliente
    purchased_by_cust = {cid: set(g["item_id"].unique()) for cid, g in df.groupby("customer_id")}

    neg_rows = []

    # ✅ FIX: iteramos por filas y accedemos por nombre
    for _, r in cust_stats.iterrows():
        cid = r["customer_id"]
        purchased = purchased_by_cust.get(cid, set())

        fav_cat = r.get("fav_cat", None)
        mu = r.get("mean_price", np.nan)
        if pd.isna(mu):
            mu = float(df["item_price"].median())
        sigma = r.get("std_price", 0.0)
        if pd.isna(sigma) or sigma == 0:
            sigma = max(1.0, 0.15 * (mu or 1.0))

        # Candidatos: no comprados
        candidates = items_tbl[~items_tbl["item_id"].isin(purchased)].copy()
        if candidates.empty:
            continue

        # Filtros por categoría y precio (con relajación progresiva)
        price_low, price_high = (mu - 1.0 * sigma), (mu + 1.0 * sigma)
        if fav_cat is not None:
            cand = candidates[(candidates["item_category"] == fav_cat) & (candidates["item_price"].between(price_low, price_high))]
        else:
            cand = candidates[candidates["item_price"].between(price_low, price_high)]
        if cand.empty:
            cand = candidates[candidates["item_price"].between(mu - 2.0 * sigma, mu + 2.0 * sigma)]
        if cand.empty:
            cand = candidates

        k = min(n_per_positive, len(cand))
        if k <= 0:
            continue
        sampled = cand.sample(k, random_state=int(rng.integers(0, 1_000_000)), replace=False).copy()

        # Enriquecimiento de evento
        sampled["customer_id"] = cid

        ts = r.get("purchase_timestamp", pd.NaT)
        if pd.isna(ts):
            if "item_release_date" in sampled.columns:
                sampled["purchase_timestamp"] = _coerce_dt(sampled["item_release_date"]) + pd.to_timedelta(7, unit="D")
            else:
                sampled["purchase_timestamp"] = pd.NaT
        else:
            sampled["purchase_timestamp"] = ts

        civ = r.get("customer_item_views", np.nan)
        sampled["customer_item_views"] = civ if not pd.isna(civ) else global_med_views

        dev = r.get("purchase_device", None)
        sampled["purchase_device"] = dev if isinstance(dev, str) and len(dev) > 0 else global_device

        if "purchase_item_rating" in df.columns:
            sampled["purchase_item_rating"] = np.nan  # No marcador

        # Campos de cliente requeridos por el dataset
        for c in ["customer_gender", "customer_date_of_birth", "customer_signup_date"]:
            if c in df.columns and c not in sampled.columns:
                val_series = df.loc[df["customer_id"] == cid, c]
                sampled[c] = val_series.iloc[-1] if not val_series.empty else np.nan

        sampled["label"] = 0

        expected = [
            "purchase_id","customer_id","customer_date_of_birth","customer_gender","customer_signup_date",
            "item_id","item_title","item_category","item_price","item_img_filename",
            "item_avg_rating","item_num_ratings","item_release_date",
            "purchase_timestamp","customer_item_views","purchase_item_rating","purchase_device","label"
        ]
        for col in expected:
            if col not in sampled.columns:
                sampled[col] = np.nan

        sampled["purchase_id"] = np.nan  # nunca crear IDs sintéticos

        neg_rows.append(sampled[expected])

    if not neg_rows:
        cols = [
            "purchase_id","customer_id","customer_date_of_birth","customer_gender","customer_signup_date",
            "item_id","item_title","item_category","item_price","item_img_filename",
            "item_avg_rating","item_num_ratings","item_release_date",
            "purchase_timestamp","customer_item_views","purchase_item_rating","purchase_device","label"
        ]
        return pd.DataFrame(columns=cols)

    neg_df = pd.concat(neg_rows, ignore_index=True)
    neg_df = neg_df.drop_duplicates(subset=["customer_id","item_id"])
    return neg_df

# Compatibilidad con el nombre que usa tu pipeline
def gen_random_negatives(df, n_per_positive=1, smart=True, random_state=42):
    return gen_realistic_negatives(df, n_per_positive=n_per_positive, random_state=random_state)

if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    neg = gen_realistic_negatives(train_df, n_per_positive=2, random_state=42)
    print("✅ Negativos generados:", len(neg))
    print(neg.head(3))
