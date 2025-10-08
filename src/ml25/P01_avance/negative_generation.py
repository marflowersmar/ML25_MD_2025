import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =========================
# Configuración de rutas
# =========================
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
LOCAL_DIR = CURRENT_FILE.parent  # src/ml25/P01_avance
DATASETS_DIR = LOCAL_DIR.parent / "datasets" / "customer_purchases"  # src/ml25/datasets/customer_purchases

# =========================
# Utilidades
# =========================
def read_csv_from_datasets(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS_DIR / f"{filename}.csv")

def _numeric_id(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    extr = s.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(extr, errors="coerce")
    nodigit = s[extr.isna()]
    if not nodigit.empty:
        uniq = sorted(nodigit.unique())
        mp = {v: i + 1 for i, v in enumerate(uniq)}
        out.loc[extr.isna()] = nodigit.map(mp).astype(float)
    return out.fillna(-1).astype(int)

def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    return data

# =========================
# Perfiles (100% numéricos)
# =========================
def build_product_profile_numeric(df_pos: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
    prod = df_pos.drop_duplicates("item_id").copy()
    prod["item_id_num"] = _numeric_id(prod["item_id"])
    prod["item_category_code"] = prod["item_category"].astype(str).str.lower().astype("category").cat.codes.astype(int)

    rel = pd.to_datetime(prod["item_release_date"], errors="coerce")
    prod["days_since_release"] = (pd.Timestamp(DATA_COLLECTED_AT) - rel).dt.days

    keep_cols = [
        "item_id_num",
        "item_price",
        "item_avg_rating",
        "item_num_ratings",
        "item_category_code",
        "days_since_release",
    ]
    for c in keep_cols:
        if c not in prod.columns:
            prod[c] = 0

    prod_num = prod[keep_cols].copy()
    prod_num = _to_numeric_df(prod_num)

    if save_csv:
        out_path = LOCAL_DIR / "product_profile.csv"
        prod_num.to_csv(out_path, index=False)
    return prod_num

def load_customer_features_numeric() -> pd.DataFrame:
    cf_path = LOCAL_DIR / "customer_features.csv"
    cf = pd.read_csv(cf_path)
    if "customer_id" not in cf.columns:
        raise ValueError("customer_features.csv no contiene la columna 'customer_id'.")
    cf["customer_id"] = pd.to_numeric(cf["customer_id"], errors="coerce").fillna(0).astype(int)
    cf = _to_numeric_df(cf)
    return cf

# =========================
# Generación de negativos
# =========================
def gen_smart_negatives(df_pos: pd.DataFrame, n_per_positive: int = 3, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    cat_pref = (
        df_pos.groupby("customer_id")["item_category"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    )
    cat_to_items = df_pos.groupby("item_category")["item_id"].unique().to_dict()
    pos_counts = df_pos.groupby("customer_id")["item_id"].nunique().to_dict()

    negatives = []
    for customer in df_pos["customer_id"].unique():
        pref = cat_pref.get(customer)
        if pref is None or pref not in cat_to_items:
            continue

        candidate_items = np.array(cat_to_items[pref])
        purchased = df_pos.loc[df_pos["customer_id"] == customer, "item_id"].unique()
        non_purchased = np.setdiff1d(candidate_items, purchased)

        if len(non_purchased) == 0:
            continue

        target_k = max(1, n_per_positive * int(pos_counts.get(customer, 1)))
        k = min(target_k, len(non_purchased))
        sampled = np.random.choice(non_purchased, size=k, replace=False)

        for it in sampled:
            negatives.append({"customer_id": customer, "item_id": it, "label": 0})

    neg_df = pd.DataFrame(negatives).drop_duplicates(ignore_index=True)
    return neg_df

# =========================
# Ensamble final numérico
# =========================
def gen_final_dataset_numeric(df_pos: pd.DataFrame, df_neg: pd.DataFrame) -> pd.DataFrame:
    pos_pairs = df_pos[["customer_id", "item_id"]].copy()
    pos_pairs["label"] = 1

    pairs = pd.concat([pos_pairs, df_neg[["customer_id", "item_id", "label"]]], ignore_index=True)
    pairs = pairs.sample(frac=1, random_state=42).reset_index(drop=True)

    pairs["customer_id_num"] = _numeric_id(pairs["customer_id"])
    pairs["item_id_num"] = _numeric_id(pairs["item_id"])

    customer_feat = load_customer_features_numeric()  # ya 100% numérico
    product_feat = build_product_profile_numeric(df_pos, save_csv=True)  # también 100% numérico

    full = pairs.merge(customer_feat, left_on="customer_id_num", right_on="customer_id", how="left", suffixes=("", "_cust"))
    full = full.merge(product_feat, on="item_id_num", how="left", suffixes=("", "_prod"))

    drop_cols = ["customer_id", "item_id"]
    full = full.drop(columns=[c for c in drop_cols if c in full.columns], errors="ignore")

    full = _to_numeric_df(full)

    cols_front = ["customer_id_num", "item_id_num"]
    cols_end = ["label"]
    middle = [c for c in full.columns if c not in cols_front + cols_end]
    ordered = cols_front + middle + cols_end
    full = full[ordered]
    return full

# =========================
# Main
# =========================
if __name__ == "__main__":
    df_pos = read_csv_from_datasets("customer_purchases_train")

    df_neg = gen_smart_negatives(df_pos, n_per_positive=2, seed=42)

    train_df_full = gen_final_dataset_numeric(df_pos, df_neg)

    out_path = LOCAL_DIR / "train_df_full.csv"
    train_df_full.to_csv(out_path, index=False)
    print(f"df saved to {out_path}")
