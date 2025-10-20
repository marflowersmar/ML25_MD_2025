# negative_generation.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Set, Tuple

# ------------------------------------------------------------
# RUTAS PORTABLES Y COMPATIBLES CON TU data_processing.py
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()

# 1) Intentar usar la misma variable de entorno que usas en data_processing.py
_env_dir = os.getenv("CUSTOMER_PURCHASES_DIR", "")

if _env_dir:
    DATA_DIR = Path(_env_dir).resolve()
else:
    # 2) Fallback relativo a tu layout:
    # src/ml25/P01_avance -> subir dos niveles -> datasets/customer_purchases
    DATA_DIR = (CURRENT_FILE.parent.parent / "datasets" / "customer_purchases").resolve()

# ------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------
def _ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la ruta: {path}\n"
            f"Verifica que el dataset esté en {DATA_DIR}\n"
            f"o define CUSTOMER_PURCHASES_DIR como variable de entorno."
        )

def read_csv(filename: str) -> pd.DataFrame:
    """
    Lee CSV desde DATA_DIR conservando el mismo contrato que data_processing.py.
    """
    file = DATA_DIR / f"{filename}.csv"
    _ensure_exists(file)
    return pd.read_csv(file)

def _price_bin(x: float) -> str:
    if pd.isna(x): return "p_nan"
    if x < 200:   return "p_0_200"
    if x < 500:   return "p_200_500"
    if x < 1000:  return "p_500_1000"
    return "p_1000_plus"

# ------------------------------------------------------------
# CÁLCULOS BASE DE NEGATIVOS
# ------------------------------------------------------------
def get_negatives(df: pd.DataFrame) -> Dict[int | str, Set[int | str]]:
    """
    Para cada customer_id devuelve el conjunto de items NO comprados.
    No usa columnas de label para evitar fugas.
    """
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())

    negatives: Dict[int | str, Set[int | str]] = {}
    for customer in unique_customers:
        purchased_items = df.loc[df["customer_id"] == customer, "item_id"].unique()
        non_purchased = unique_items - set(purchased_items)
        negatives[customer] = non_purchased
    return negatives

def gen_all_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera el producto cartesiano customer x items no comprados.
    Útil para depuración, no recomendado para entrenar por tamaño.
    """
    negatives = get_negatives(df)
    rows: List[dict] = []
    for customer_id, item_set in negatives.items():
        if not item_set:
            continue
        rows.extend({"customer_id": customer_id, "item_id": it, "label": 0} for it in item_set)
    return pd.DataFrame(rows, columns=["customer_id", "item_id", "label"])

# ------------------------------------------------------------
# MODO SMART PARA SAMPLING NEGATIVO
# ------------------------------------------------------------
def _prepare_item_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla por item con features mínimas para muestreo smart
    sin requerir más columnas que luego se enriquecen en data_processing.py.
    """
    keep_cols = ["item_id", "item_category", "item_price", "item_title"]
    missing = [c for c in keep_cols if c not in df.columns]
    # Asegurar columnas básicas
    tmp = df.copy()
    for c in missing:
        tmp[c] = np.nan

    item_tbl = (
        tmp[keep_cols]
        .drop_duplicates(subset=["item_id"])
        .reset_index(drop=True)
    )

    # Buckets de precio para similitud básica
    item_tbl["price_bin"] = item_tbl["item_price"].apply(_price_bin)
    # Normalizar categoría de texto
    item_tbl["item_category"] = item_tbl["item_category"].fillna("unknown_category").astype(str)
    return item_tbl

def _customer_preferences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preferencias simples por customer: categoría más frecuente y bin de precio medio.
    Evita usar cualquier información del futuro del test.
    """
    tmp = df.copy()
    tmp["item_category"] = tmp["item_category"].fillna("unknown_category").astype(str)

    # Categoría favorita
    fav_cat = (
        tmp.groupby(["customer_id", "item_category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["customer_id", "cnt"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .rename(columns={"item_category": "fav_category"})
        [["customer_id", "fav_category"]]
    )

    # Bin de precio típico
    mean_price = tmp.groupby("customer_id")["item_price"].mean().reset_index(name="mean_price")
    mean_price["fav_price_bin"] = mean_price["mean_price"].apply(_price_bin)

    prefs = fav_cat.merge(mean_price[["customer_id", "fav_price_bin"]], on="customer_id", how="left")
    return prefs

def gen_smart_negatives(df: pd.DataFrame,
                        n_per_positive: int = 1,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Muestreo negativo que respeta afinidad por:
      1) categoría más frecuente del cliente
      2) bin de precio típico del cliente
    Genera hasta n_per_positive items no comprados por cada compra positiva.
    """
    rng = np.random.default_rng(seed=random_state)
    item_tbl = _prepare_item_index(df)
    prefs = _customer_preferences(df)

    # Índices por clave para acelerar el muestreo
    by_cat = {cat: set(g["item_id"].tolist()) for cat, g in item_tbl.groupby("item_category")}
    by_price = {pb: set(g["item_id"].tolist()) for pb, g in item_tbl.groupby("price_bin")}
    all_items = set(item_tbl["item_id"].tolist())

    # Historial por cliente para excluir comprados
    bought_by_cust = df.groupby("customer_id")["item_id"].apply(set).to_dict()

    # Recorrer positivos como referencia de conteo
    positives = df[["customer_id", "item_id"]].copy()
    rows: List[dict] = []

    for idx, row in positives.iterrows():
        c = row["customer_id"]
        # Candidatos no comprados
        pool = all_items - bought_by_cust.get(c, set())
        if not pool:
            continue

        # Afinidad por preferencias del cliente
        pref_row = prefs[prefs["customer_id"] == c]
        if not pref_row.empty:
            fav_cat = pref_row.iloc[0]["fav_category"]
            fav_bin = pref_row.iloc[0]["fav_price_bin"]
            set_cat = by_cat.get(fav_cat, set())
            set_bin = by_price.get(fav_bin, set())
            # Intersección preferida
            pref_pool = (set_cat & set_bin) & pool
            # Si queda vacío, intentar por categoría
            if not pref_pool:
                pref_pool = set_cat & pool
            # Si sigue vacío, intentar por bin de precio
            if not pref_pool:
                pref_pool = set_bin & pool
            # Si aún vacío, usar pool completo
            if not pref_pool:
                pref_pool = pool
        else:
            pref_pool = pool

        k = min(n_per_positive, len(pref_pool))
        if k <= 0:
            continue
        sampled = rng.choice(list(pref_pool), size=k, replace=False)
        rows.extend({"customer_id": c, "item_id": it, "label": 0} for it in sampled)

    if not rows:
        return pd.DataFrame(columns=["customer_id", "item_id", "label"])
    return pd.DataFrame(rows, columns=["customer_id", "item_id", "label"])

# ------------------------------------------------------------
# INTERFAZ PRINCIPAL USADA POR TU data_processing.py
# ------------------------------------------------------------
def gen_random_negatives(df: pd.DataFrame,
                         n_per_positive: int = 1,
                         smart: bool = True,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Punto único llamado desde data_processing.build_training_table(...).
    - smart=True usa gen_smart_negatives con afinidad por categoría y price_bin.
    - smart=False hace muestreo uniforme sin afinidad.
    En ambos casos se evita muestrear items ya comprados por el cliente.
    """
    if smart:
        return gen_smart_negatives(df, n_per_positive=n_per_positive, random_state=random_state)

    # Uniforme
    rng = np.random.default_rng(seed=random_state)
    all_items = set(df["item_id"].unique())
    bought_by_cust = df.groupby("customer_id")["item_id"].apply(set).to_dict()
    positives = df[["customer_id", "item_id"]].copy()

    rows: List[dict] = []
    for _, r in positives.iterrows():
        c = r["customer_id"]
        pool = list(all_items - bought_by_cust.get(c, set()))
        if not pool:
            continue
        k = min(n_per_positive, len(pool))
        sampled = rng.choice(pool, size=k, replace=False)
        rows.extend({"customer_id": c, "item_id": it, "label": 0} for it in sampled)

    if not rows:
        return pd.DataFrame(columns=["customer_id", "item_id", "label"])
    return pd.DataFrame(rows, columns=["customer_id", "item_id", "label"])

# ------------------------------------------------------------
# OPCIONAL: GENERAR DATASET FINAL CON NEGATIVOS
# No lo usa tu build_training_table, pero lo dejamos funcional.
# ------------------------------------------------------------
def gen_final_dataset(train_df: pd.DataFrame, negatives: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un dataframe con positivos de train_df y los negativos recibidos,
    con las mismas columnas base de train_df y una columna label.
    Evita duplicar columnas ID innecesarias.
    """
    # Positivos con label=1
    pos = train_df.copy()
    pos = pos.assign(label=1)

    # Asegurar columnas mínimas en negatives
    if "customer_id" not in negatives.columns or "item_id" not in negatives.columns:
        raise ValueError("negatives debe contener columnas customer_id e item_id")

    # Reducir train_df a columnas de ítem y de customer necesarias para enriquecer
    item_cols = [
        "item_id", "item_title", "item_category", "item_price",
        "item_img_filename", "item_avg_rating", "item_num_ratings", "item_release_date"
    ]
    cust_cols = ["customer_id", "customer_date_of_birth", "customer_gender", "customer_signup_date"]
    purch_cols = ["purchase_id", "purchase_timestamp", "customer_item_views",
                  "purchase_item_rating", "purchase_device"]

    # Tablas únicas para merge
    item_tbl = train_df[item_cols].drop_duplicates("item_id")
    cust_tbl = train_df[cust_cols].drop_duplicates("customer_id")

    neg = negatives.merge(cust_tbl, on="customer_id", how="left").merge(item_tbl, on="item_id", how="left")
    # La compra no existe para negativos
    for c in purch_cols:
        neg[c] = np.nan
    neg["label"] = 0

    # Unificar columnas
    all_cols = list(pos.columns.union(neg.columns))
    pos = pos.reindex(columns=all_cols)
    neg = neg.reindex(columns=all_cols)

    full = pd.concat([pos, neg], ignore_index=True)
    # Barajar sin usar columnas de tiempo
    full = full.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return full

# ------------------------------------------------------------
# PRUEBA LOCAL
# ------------------------------------------------------------
if __name__ == "__main__":
    print(f"Usando DATA_DIR = {DATA_DIR}")
    train_df = read_csv("customer_purchases_train")

    # Sanity checks
    allnegs = gen_all_negatives(train_df)
    print("allnegs:", allnegs.shape)

    randnegs_uniform = gen_random_negatives(train_df, n_per_positive=2, smart=False, random_state=123)
    print("randnegs_uniform:", randnegs_uniform.shape, randnegs_uniform.head())

    randnegs_smart = gen_random_negatives(train_df, n_per_positive=2, smart=True, random_state=123)
    print("randnegs_smart:", randnegs_smart.shape, randnegs_smart.head())

    # Dataset final opcional
    df_final = gen_final_dataset(train_df, randnegs_smart)
    print("final_dataset:", df_final.shape)
