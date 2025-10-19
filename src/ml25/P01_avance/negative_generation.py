import numpy as np
import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path(r"C:\Users\fomm0\OneDrive\Documents\ICE V\APRENDIZAJE DE MAQUINA\ML25_MD_2025\src\ml25\datasets\customer_purchases")

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    return pd.read_csv(file)

def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())
    negatives = {}
    for customer in unique_customers:
        purchased = set(df.loc[df["customer_id"] == customer, "item_id"])
        non_purchased = list(unique_items - purchased)
        negatives[customer] = non_purchased
    return negatives

def gen_all_negatives(df):
    """
    Genera TODOS los pares cliente–item no comprados (enorme). Enriquecidos con atributos de item.
    """
    items = df[["item_id", "item_title", "item_category", "item_price", "item_img_filename"]].drop_duplicates()
    negatives = get_negatives(df)
    rows = []
    for cust, item_list in negatives.items():
        if not item_list:
            continue
        subset = items[items["item_id"].isin(item_list)]
        if subset.empty:
            continue
        tmp = subset.copy()
        tmp.insert(0, "customer_id", cust)
        tmp["label"] = 0
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["customer_id","item_id","item_title","item_category","item_price","item_img_filename","label"])
    neg_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["customer_id","item_id"])
    return neg_df

def gen_random_negatives(df, n_per_positive=1, smart=True, random_state=42):
    """
    Genera negativos realistas y ENRIQUECIDOS con atributos de item.
    - n_per_positive: número de negativos por compra positiva del cliente
    - smart=True: filtra por categoría favorita y rango de precio típico del cliente
    """
    rng = np.random.default_rng(random_state)
    # Mapa de items (para enriquecer columnas y evitar NaN “delatores”)
    items = df[["item_id", "item_title", "item_category", "item_price", "item_img_filename"]].drop_duplicates()

    customers = df["customer_id"].unique()
    negatives_list = []

    for cust in customers:
        cust_hist = df[df["customer_id"] == cust]
        purchased = set(cust_hist["item_id"])

        fav_cat = cust_hist["item_category"].mode()[0] if not cust_hist["item_category"].mode().empty else None
        mean_price = cust_hist["item_price"].mean()
        std_price = cust_hist["item_price"].std()
        if pd.isna(std_price) or std_price == 0:
            std_price = max(mean_price * 0.1, 1.0)

        # Candidatos
        candidates = items[~items["item_id"].isin(purchased)].copy()
        if smart and fav_cat is not None:
            price_low, price_high = mean_price - std_price, mean_price + std_price
            candidates = candidates[
                (candidates["item_category"] == fav_cat) &
                (candidates["item_price"].between(price_low, price_high))
            ]
            if candidates.empty:
                candidates = items[~items["item_id"].isin(purchased)].copy()

        if candidates.empty:
            continue

        n_pos = len(cust_hist)
        n_to_sample = min(n_per_positive * n_pos, len(candidates))
        sampled = candidates.sample(n_to_sample, replace=False, random_state=random_state)

        sampled = sampled.copy()
        sampled.insert(0, "customer_id", cust)
        sampled["label"] = 0
        negatives_list.append(sampled)

    if not negatives_list:
        return pd.DataFrame(columns=["customer_id","item_id","item_title","item_category","item_price","item_img_filename","label"])

    neg_df = pd.concat(negatives_list, ignore_index=True)
    neg_df = neg_df.drop_duplicates(subset=["customer_id", "item_id"])
    return neg_df

# Prueba local
if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    neg = gen_random_negatives(train_df, n_per_positive=1, smart=True)
    print("✅ Negativos generados:", len(neg))
    print(neg.head())
