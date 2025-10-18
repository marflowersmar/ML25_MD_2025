# data_processing.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import sparse

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
import joblib

# ✅ IMPORT LOCAL: negative_generation.py está en la MISMA carpeta
from negative_generation import (
    gen_random_negatives,
    # gen_all_negatives,  # opcional si la usas
)

# ================== CONFIG ==================
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
# datasets en: .../src/ml25/datasets/customer_purchases
DATA_DIR = (CURRENT_FILE.parent / "../datasets/customer_purchases").resolve()
ENC_DIR = (DATA_DIR / "encoders"); ENC_DIR.mkdir(exist_ok=True, parents=True)
PREPROCESSOR_PATH = DATA_DIR / "preprocessor.pkl"
TODAY = datetime.combine(DATA_COLLECTED_AT, datetime.min.time())

# ====== COLOR MAP (train) -> códigos fijos 1..8 ======
# Orden solicitado: blue, black, green, orange, pink, red, white, yellow
IMG_COLOR_CODE = {
    "imgbl.jpg": 1,  # blue
    "imgb.jpg":  2,  # black
    "imgg.jpg":  3,  # green
    "imgo.jpg":  4,  # orange
    "imgp.jpg":  5,  # pink
    "imgr.jpg":  6,  # red
    "imgw.jpg":  7,  # white
    "imgy.jpg":  8,  # yellow
}

# ================== I/O ==================
def read_csv(filename: str):
    file = str(DATA_DIR / f"{filename}.csv")
    df = pd.read_csv(file)
    return df

def save_df(df, filename: str):
    save_path = str(DATA_DIR / filename)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")

def df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

# ================== ENCODERS PERSISTENTES ==================
def _save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _fit_or_load_label_codes(values: pd.Series, name: str) -> dict:
    path = ENC_DIR / f"{name}_codes.json"
    if path.exists():
        return {k: int(v) for k, v in _load_json(path).items()}
    uniq = sorted([str(x) for x in pd.Series(values).dropna().unique()])
    codes = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reservado para unknown
    _save_json(codes, path)
    return codes

def _fit_or_load_price_bins(prices: pd.Series, nbins: int = 4):
    path = ENC_DIR / "price_bins.json"
    if path.exists():
        data = _load_json(path)
        return np.array(data["edges"], dtype=float)
    s = pd.to_numeric(prices, errors="coerce").clip(lower=0).dropna()
    if s.nunique() < 2:
        edges = np.linspace(float(s.min() if len(s) else 0), float(s.max() if len(s) else 1), nbins + 1)
    else:
        _, bins = pd.qcut(s, q=nbins, retbins=True, duplicates="drop")
        edges = bins
    _save_json({"edges": list(map(float, edges))}, path)
    return np.array(edges, dtype=float)

def _bin_code_from_edges(x: float, edges: np.ndarray) -> int:
    if pd.isna(x):
        return 0
    return int(np.digitize([x], edges[1:-1], right=True)[0] + 1)  # 1..nbins

def _safe_mode(s: pd.Series):
    s = pd.Series(s).dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else np.nan

def _days_between_mean(ts: pd.Series) -> float:
    ts = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if ts.size < 2:
        return -1.0
    diffs = ts.diff().dropna().dt.days.astype(float)
    return float(diffs.mean()) if not diffs.empty else -1.0

def _count_token_in_text(series: pd.Series, tokens) -> int:
    if series is None or series.isna().all():
        return 0
    text = series.astype(str).str.lower()
    pat = "|".join([pd.regex.re.escape(t) for t in tokens])
    return int(text.str.contains(pat, regex=True).sum())

# ================== CUSTOMER FEATURES ==================
def extract_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera customer_features.csv con SOLO columnas numéricas (incluyendo customer_id).
    Requiere en df: customer_id, customer_date_of_birth, customer_gender, customer_signup_date,
                    item_price, item_category, item_title, purchase_timestamp,
                    purchase_device, item_img_filename
    """
    data = df.copy()

    # Fechas
    data["customer_date_of_birth"] = pd.to_datetime(data.get("customer_date_of_birth"), errors="coerce")
    data["customer_signup_date"]   = pd.to_datetime(data.get("customer_signup_date"), errors="coerce")
    data["purchase_timestamp"]     = pd.to_datetime(data.get("purchase_timestamp"), errors="coerce")

    # Mapeos persistentes (IDs, categorías, dispositivos)
    id_codes  = _fit_or_load_label_codes(data["customer_id"], "customer_id")
    cat_codes = _fit_or_load_label_codes(data.get("item_category"), "item_category")
    dev_codes = _fit_or_load_label_codes(data.get("purchase_device"), "purchase_device")
    edges     = _fit_or_load_price_bins(data.get("item_price"))

    # Customer ID numérico (sin columna adicional)
    data["customer_id"] = data["customer_id"].astype(str).map(lambda x: id_codes.get(x, 0)).astype(int)

    # Género codificado
    gender_map = {"F": 1, "M": 2, "f": 1, "m": 2}
    gender_code = data.get("customer_gender", pd.Series(index=data.index, dtype=object)).astype(str).map(gender_map).fillna(0).astype(int)

    # Color del producto (desde filename) → código
    data["item_color_code"] = data.get("item_img_filename", pd.Series(index=data.index, dtype=object)).map(IMG_COLOR_CODE).fillna(0).astype(int)

    # Group by cliente
    grp = data.groupby("customer_id", dropna=False)

    records = []
    for cust_id, subdf in grp:
        prices = pd.to_numeric(subdf.get("item_price"), errors="coerce")
        total_compras  = int(subdf.shape[0])
        gasto_prom     = float(prices.mean()) if total_compras > 0 else 0.0
        max_price      = float(prices.max()) if total_compras > 0 else 0.0
        min_price      = float(prices.min()) if total_compras > 0 else 0.0

        # Edad / antigüedad
        dob = subdf["customer_date_of_birth"].dropna()
        edad = int(((TODAY - dob.iloc[0]).days // 365)) if not dob.empty else -1
        sdate = subdf["customer_signup_date"].dropna()
        antiguedad = int((TODAY - sdate.iloc[0]).days) if not sdate.empty else -1

        # Gender code (primer no-nulo)
        g = gender_code.loc[subdf.index]
        gender_c = int(g[g != 0].iloc[0]) if (g != 0).any() else 0

        # Categoría / dispositivo frecuentes (codificados)
        cat_mode = _safe_mode(subdf.get("item_category"))
        cat_code = int(cat_codes.get(str(cat_mode), 0))
        dev_mode = _safe_mode(subdf.get("purchase_device"))
        dev_code = int(dev_codes.get(str(dev_mode), 0))

        # Bin de precio preferido (modo del bin)
        price_bins = prices.map(lambda x: _bin_code_from_edges(x, edges))
        pref_price_code = int(_safe_mode(price_bins))

        # Días promedio entre compras y recency
        dias_promedio = _days_between_mean(subdf.get("purchase_timestamp"))
        last_ts = subdf["purchase_timestamp"].dropna()
        recency = int((TODAY - last_ts.max()).days) if not last_ts.empty else -1

        # Contadores de color por título (inglés/español)
        title_series = subdf.get("item_title")
        color_vocab = {
            "black": ["black", "negro", "negra"],
            "blue":  ["blue", "azul"],
            "green": ["green", "verde"],
            "orange":["orange", "naranja"],
            "pink":  ["pink", "rosa", "fucsia"],
            "red":   ["red", "rojo", "roja"],
            "white": ["white", "blanco", "blanca"],
            "yellow":["yellow", "amarillo", "amarilla"],
        }
        color_counts = {
            f"customer_color_{k}_count": _count_token_in_text(title_series, toks)
            for k, toks in color_vocab.items()
        }

        # Contadores por categoría
        cat_series = subdf.get("item_category").astype(str).str.lower()
        cat_targets = {
            "blouse": "blouse", "dress": "dress", "jacket": "jacket", "jeans": "jeans",
            "shirt": "shirt", "shoes": "shoes", "skirt": "skirt", "slacks": "slacks",
            "suit": "suit", "t-shirt": ["t-shirt", "tshirt", "tee"],
        }
        cat_counts = {}
        for key, patt in cat_targets.items():
            if isinstance(patt, list):
                patt_regex = "|".join([pd.regex.re.escape(p) for p in patt])
                cnt = int(cat_series.str.contains(patt_regex, regex=True).sum())
            else:
                cnt = int((cat_series == patt).sum())
            cat_counts[f"customer_cat_{key}_count"] = cnt

        # Color code más frecuente por imagen
        color_mode = _safe_mode(subdf.get("item_color_code"))
        color_mode = int(color_mode) if not pd.isna(color_mode) else 0

        rec = {
            "customer_id": int(cust_id),
            "customer_edad": edad,
            "customer_gender_code": gender_c,
            "customer_total_compras": total_compras,
            "customer_gasto_promedio": gasto_prom,
            "customer_antiguedad_dias": antiguedad,
            "customer_categoria_frecuente_code": cat_code,
            "customer_device_frecuente_code": dev_code,
            "customer_preferred_price_bin_code": pref_price_code,
            "customer_max_item_price": max_price,
            "customer_min_item_price": min_price,
            "customer_dias_promedio_compra": float(dias_promedio),
            "customer_recency_dias": recency,
            "customer_color_code_frecuente": color_mode,
        }
        rec.update(color_counts)
        rec.update(cat_counts)
        records.append(rec)

    customer_feat = pd.DataFrame.from_records(records)

    # Todo numérico
    customer_feat = customer_feat.apply(pd.to_numeric, errors="coerce").fillna(0)
    # Guardar CSV
    save_path = DATA_DIR / "customer_features.csv"
    customer_feat.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")
    return customer_feat

# ================== PREPROCESSING MODEL ==================
def build_processor(df, numerical_features, categorical_features, free_text_features, training=True):
    """
    Ajusta o carga un ColumnTransformer y devuelve un DataFrame transformado.
    """
    if training:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        free_text_transformers = []
        for col in free_text_features:
            free_text_transformers.append((col, CountVectorizer(), col))

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
                *free_text_transformers,
            ],
            remainder="passthrough",
            sparse_threshold=1.0,
        )
        df_fit = df.drop(columns=["label"], errors="ignore")
        processed_array = preprocessor.fit_transform(df_fit)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
    else:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        processed_array = preprocessor.transform(df)

    if sparse.issparse(processed_array):
        processed_array = processed_array.toarray()

    # ---- Nombres de columnas ----
    num_cols = list(numerical_features)
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)

    bow_cols = []
    for col in free_text_features:
        vec = preprocessor.named_transformers_[col]
        bow_cols.extend([f"{col}_bow_{t}" for t in vec.get_feature_names_out()])

    other_cols = [
        c for c in df.columns
        if c not in list(numerical_features) + list(categorical_features) + list(free_text_features)
    ]

    final_cols = list(num_cols) + list(cat_cols) + list(bow_cols) + list(other_cols)
    processed_df = pd.DataFrame(processed_array, columns=final_cols)
    return processed_df

def preprocess(raw_df, training=False):
    """
    Preprocesamiento: crea columnas auxiliares y aplica el ColumnTransformer.
    """
    # Flag: categoría preferida
    raw_df["customer_cat_is_prefered"] = (raw_df["item_category"] == raw_df["customer_prefered_cat"]).astype(int)

    # Color code numérico si viene filename
    if "item_img_filename" in raw_df.columns and "item_color_code" not in raw_df.columns:
        raw_df["item_color_code"] = raw_df["item_img_filename"].map(IMG_COLOR_CODE).fillna(0).astype(int)

    numerical_feat = [
        "item_price",
        "customer_age_years",
        "customer_tenure_years",
        "item_color_code",
    ]
    categorical_features = ["customer_gender", "item_category"]
    free_text_features = ["item_title"]

    processed_df = build_processor(
        raw_df,
        numerical_feat,
        categorical_features,
        free_text_features,
        training=training,
    )

    dropcols = [
        "purchase_id",
        "customer_date_of_birth",
        "customer_signup_date",
        "purchase_item_rating",
        "purchase_device",
        "purchase_timestamp",
        "customer_item_views",
        "item_release_date",
        "item_avg_rating",
        "item_num_ratings",
        "customer_prefered_cat",
        "item_img_filename",
    ]
    processed_df = processed_df.drop(columns=[c for c in dropcols if c in processed_df.columns], errors="ignore")
    return processed_df

# ================== DATA READERS ==================
def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)

    # Negativos 1:1
    train_df_neg = gen_random_negatives(train_df, n_per_positive=1)
    train_df_neg = train_df_neg.drop_duplicates(subset=["customer_id", "item_id"])

    # Enriquecer positivos con features de cliente
    train_df_cust = pd.merge(train_df, customer_feat, on="customer_id", how="left")

    processed_pos = preprocess(train_df_cust, training=True)
    processed_pos["label"] = 1

    all_columns = processed_pos.columns

    # Features de item
    item_feat = [col for col in all_columns if "item" in col] + ["item_id"]
    unique_items = processed_pos[item_feat].drop_duplicates(subset=["item_id"])

    # Features de cliente
    customer_cols = [col for col in all_columns if "customer_" in col] + ["customer_id"]
    unique_customers = processed_pos[customer_cols].drop_duplicates(subset=["customer_id"])

    # Enriquecer negativos
    processed_neg = pd.merge(train_df_neg, unique_items, on="item_id", how="left")
    processed_neg = pd.merge(processed_neg, unique_customers, on="customer_id", how="left")
    processed_neg["label"] = 0

    processed_full = (
        pd.concat([processed_pos, processed_neg], axis=0)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    shuffled = df_to_numeric(processed_full)
    y = shuffled["label"]
    X = shuffled.drop(columns=["label", "customer_id", "item_id"], errors="ignore")
    return X, y

def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")  # generado en train

    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")

    if "item_img_filename" in merged.columns and "item_color_code" not in merged.columns:
        merged["item_color_code"] = merged["item_img_filename"].map(IMG_COLOR_CODE).fillna(0).astype(int)

    processed = preprocess(merged, training=False)
    processed = processed.drop(columns=[], errors="ignore")
    return df_to_numeric(processed)

# ================== MAIN ==================
if __name__ == "__main__":
    X_train, y_train = read_train_data()
    print(X_train.info())
    test_df = read_csv("customer_purchases_test")
    X_test = read_test_data()
    print(test_df.columns)
    print("hola")
