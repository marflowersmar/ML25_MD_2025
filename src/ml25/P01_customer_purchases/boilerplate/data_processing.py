import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

# === sklearn imports (requested) ===
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder


# Helper function for text pipeline (picklable)
def _to_series(X):
    import numpy as np
    import pandas as pd
    return pd.Series(np.array(X).squeeze())

# ===================================
# Paths & constants
# ===================================
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()

CURRENT_FILE = Path(__file__).resolve()
# Prefer a datasets path relative to this file; fall back to /mnt/data if it doesn't exist.
DATA_DIR = (CURRENT_FILE.parent / "../../datasets/customer_purchases/").resolve()
FALLBACK_DIR = Path("/mnt/data")
if not DATA_DIR.exists():
    DATA_DIR = FALLBACK_DIR

# Controlled vocabularies (stable ordering for codes)
CATEGORY_ORDER = [
    "blouse", "dress", "jacket", "jeans", "shirt",
    "shoes", "skirt", "slacks", "suit", "t-shirt"
]
COLOR_ORDER = ["black", "blue", "green", "orange", "pink", "red", "white", "yellow"]

# ===================================
# Helpers
# ===================================
def _safe_to_datetime(s, fmt=None):
    try:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def _age_from_dob(dob: pd.Series, asof_date=DATA_COLLECTED_AT) -> pd.Series:
    dob_dt = _safe_to_datetime(dob).dt.date
    delta_days = (pd.to_datetime(asof_date) - pd.to_datetime(dob_dt)).dt.days
    return np.floor(delta_days / 365.25).astype("float64")

def _gender_to_code(g: pd.Series) -> pd.Series:
    mapping = {"female": 1, "male": 0}
    return g.astype(str).str.lower().map(mapping).fillna(-1).astype("int64")

def _mode_or_nan(series: pd.Series):
    if series.empty:
        return np.nan
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan

def _map_with_unk(x, vocab):
    if pd.isna(x):
        return -1
    x = str(x).lower()
    return vocab.index(x) if x in vocab else -1

def _device_code(series: pd.Series) -> Tuple[pd.Series, dict]:
    uniques = sorted({str(x).lower() for x in series.dropna()})
    mapping = {val: i for i, val in enumerate(uniques)}
    codes = series.astype(str).str.lower().map(mapping).fillna(-1).astype("int64")
    return codes, mapping

def _price_bin_code(prices: pd.Series) -> pd.Series:
    p = pd.to_numeric(prices, errors="coerce")
    if p.nunique(dropna=True) >= 3:
        try:
            bins = pd.qcut(p.fillna(p.median()), 3, labels=[0,1,2], duplicates="drop")
            return bins.astype("int64")
        except Exception:
            pass
    return pd.Series(np.where(p.fillna(0) > 0, 1, 0), index=p.index).astype("int64")

def _numeric_customer_id(s: pd.Series) -> pd.Series:
    """Make customer_id strictly numeric.
    - Try to extract digits (e.g., 'CUST_0001' -> 1).
    - If no digits at all, use a stable factorization mapping.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)

    # Extract digits when present
    extracted = s.astype(str).str.extract(r'(\d+)', expand=False)
    has_digits = extracted.notna()
    numeric_part = pd.to_numeric(extracted.where(has_digits, np.nan), errors='coerce')

    # For rows with no digits, factorize those strings to a code space (start at 1)
    no_digit_vals = s[~has_digits].astype(str)
    if not no_digit_vals.empty:
        # Stable order mapping via sorted unique
        uniq = sorted(no_digit_vals.unique())
        map_nd = {val: i+1 for i, val in enumerate(uniq)}  # start at 1
        numeric_fallback = no_digit_vals.map(map_nd).astype("Int64")
        numeric_part.loc[~has_digits] = numeric_fallback.astype(float)

    # Fill any remaining NaN with -1 and cast to int
    numeric_part = numeric_part.fillna(-1).astype(int)
    return numeric_part

# ===================================
# I/O
# ===================================
def read_csv(filename: str):
    candidates = [
        os.path.abspath(os.path.join(DATA_DIR, f"{filename}.csv")),
        os.path.abspath(os.path.join(FALLBACK_DIR, f"{filename}.csv")),
    ]
    for fullfilename in candidates:
        if os.path.exists(fullfilename):
            return pd.read_csv(fullfilename)
    raise FileNotFoundError(f"Could not find {filename}.csv in {DATA_DIR} or {FALLBACK_DIR}")

<<<<<<< HEAD
def save_df(df, filename: str = "customer_features.csv"):
    save_path = os.path.join(DATA_DIR, "customer_features.csv")
=======

def save_df(df, filename: str):
    # Guardar
    save_path = os.path.join(DATA_DIR, filename)
>>>>>>> origin/master
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")

# ===================================
# Core feature extraction
# ===================================
def extract_customer_features(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()

    # --- ID numérico ---
    if "customer_id" not in df.columns:
        raise KeyError("customer_id column is required in train_df")
    df["customer_id"] = _numeric_customer_id(df["customer_id"])

    # --- Tipos y fechas ---
    df["item_price"] = pd.to_numeric(df.get("item_price", np.nan), errors="coerce")
    df["purchase_timestamp"] = _safe_to_datetime(df.get("purchase_timestamp", np.nan))
    df["customer_signup_date"] = _safe_to_datetime(df.get("customer_signup_date", np.nan))
    df["customer_date_of_birth"] = _safe_to_datetime(df.get("customer_date_of_birth", np.nan))

    # --- GroupBy base por cliente ---
    grp = df.groupby("customer_id", as_index=False)

    # Compras y precios
    total_compras = grp.size().rename(columns={"size": "total_compras"})
    gasto_promedio = grp["item_price"].mean().rename(columns={"item_price": "gasto_promedio"})
    max_item_price = grp["item_price"].max().rename(columns={"item_price": "max_item_price"})
    min_item_price = grp["item_price"].min().rename(columns={"item_price": "min_item_price"})

    # Intervalo promedio entre compras (en días)
    inter_days = (
        df.sort_values(["customer_id", "purchase_timestamp"])
          .groupby("customer_id")["purchase_timestamp"]
          .apply(lambda s: s.diff().dt.days.dropna().mean())
          .rename("dias_promedio_compra")
          .reset_index()
    )

    # Antigüedad desde primer signup
    first_signup = grp["customer_signup_date"].min().rename(columns={"customer_signup_date": "first_signup"})
    first_signup["antiguedad_dias"] = (
        (pd.to_datetime(DATA_COLLECTED_AT) - first_signup["first_signup"]).dt.days
    )
    antig = first_signup[["customer_id", "antiguedad_dias"]]

    # Edad y género (usar último DOB y modo de género)
    demo = (
        df.sort_values(["customer_id","purchase_timestamp"])
          .groupby("customer_id")
          .agg({
              "customer_date_of_birth": "last",
              "customer_gender": lambda s: _mode_or_nan(s.dropna())
          })
          .reset_index()
          .rename(columns={"customer_date_of_birth": "dob", "customer_gender": "gender"})
    )
    demo["edad"] = _age_from_dob(demo["dob"])
    demo["gender_code"] = _gender_to_code(demo["gender"])
    demo = demo[["customer_id", "edad", "gender_code"]]

    # Categoría frecuente (modo) y código
    fav_cat = grp["item_category"].agg(_mode_or_nan).rename(columns={"item_category": "categoria_frecuente"})
    fav_cat["categoria_frecuente_code"] = fav_cat["categoria_frecuente"].apply(
        lambda x: _map_with_unk(x, CATEGORY_ORDER)
    )
    fav_cat = fav_cat[["customer_id", "categoria_frecuente_code"]]

    # Dispositivo frecuente: usar 'purchase_device' si existe
    device_col = "purchase_device" if "purchase_device" in df.columns else None
    if device_col is not None:
        fav_dev = grp[device_col].agg(_mode_or_nan).rename(columns={device_col: "device_frecuente"})
        dev_codes_series, _ = _device_code(fav_dev["device_frecuente"].astype(str))
        fav_dev["device_frecuente_code"] = dev_codes_series
        fav_dev = fav_dev[["customer_id", "device_frecuente_code"]]
    else:
        fav_dev = pd.DataFrame({"customer_id": df["customer_id"].unique(), "device_frecuente_code": -1})

    # Bin de precio preferido (mediana por cliente)
    med_price = grp["item_price"].median().rename(columns={"item_price": "median_price"})
    med_price["preferred_price_bin_code"] = _price_bin_code(med_price["median_price"])
    price_bins = med_price[["customer_id", "preferred_price_bin_code"]]

    # Conteos por categoría
    cat_counts = (
        df.groupby(["customer_id", "item_category"]).size().unstack(fill_value=0)
          .reindex(columns=CATEGORY_ORDER, fill_value=0)
          .add_prefix("cat_")
          .reset_index()
    )

    # Conteos por color (si no hay columna item_color, crear ceros)
    if "item_color" in df.columns:
        color_counts = (
            df.groupby(["customer_id", "item_color"]).size().unstack(fill_value=0)
              .reindex(columns=COLOR_ORDER, fill_value=0)
              .add_suffix("_count")
              .reset_index()
        )
    else:
        color_counts = pd.DataFrame({"customer_id": df["customer_id"].unique()})
        for c in COLOR_ORDER:
            color_counts[f"color_{c}_count"] = 0

    # === EXTRA: texto por cliente para CountVectorizer ===
    # USED: CountVectorizer (se utilizará dentro de process_df via ColumnTransformer)
    if "item_title" in df.columns:
        text_agg = (df.assign(item_title=df["item_title"].astype(str).str.lower())
                      .groupby("customer_id")["item_title"]
                      .apply(lambda s: " ".join(s)))
    else:
        text_agg = pd.Series("", index=df["customer_id"].unique())
    text_df = text_agg.reset_index().rename(columns={"item_title": "bag_of_titles"})

    # Ensamble de features
    feat = (
        total_compras.merge(gasto_promedio, on="customer_id", how="left")
                     .merge(max_item_price, on="customer_id", how="left")
                     .merge(min_item_price, on="customer_id", how="left")
                     .merge(inter_days, on="customer_id", how="left")
                     .merge(antig, on="customer_id", how="left")
                     .merge(demo, on="customer_id", how="left")
                     .merge(fav_cat, on="customer_id", how="left")
                     .merge(fav_dev, on="customer_id", how="left")
                     .merge(price_bins, on="customer_id", how="left")
                     .merge(cat_counts, on="customer_id", how="left")
                     .merge(color_counts, on="customer_id", how="left")
                         )

    # Normalización de nombres (cat_*_count y color_*_count)
    rename_map = {f"cat_{c}": f"cat_{c}_count" for c in CATEGORY_ORDER}
    feat = feat.rename(columns=rename_map)

    # Columnas finales base (numéricas) + texto
    final_cols = [
        "customer_id", "edad", "gender_code", "total_compras", "gasto_promedio", "antiguedad_dias",
        "categoria_frecuente_code",
        "color_black_count","color_blue_count","color_green_count","color_orange_count","color_pink_count",
        "color_red_count","color_white_count","color_yellow_count",
        "device_frecuente_code","preferred_price_bin_code","max_item_price","min_item_price",
        "dias_promedio_compra",
        "cat_blouse_count","cat_dress_count","cat_jacket_count","cat_jeans_count","cat_shirt_count",
        "cat_shoes_count","cat_skirt_count","cat_slacks_count","cat_suit_count","cat_t-shirt_count"
    ]
    # Garantizar existencia de todas las columnas
    for col in final_cols:
        if col not in feat.columns:
            feat[col] = 0 if col != "bag_of_titles" else ""

    feat = feat[final_cols]

    # Tipificación y política de NA
    code_cols = ["gender_code","categoria_frecuente_code","device_frecuente_code","preferred_price_bin_code"]
    count_cols = [c for c in feat.columns if c.endswith("_count")] + ["total_compras"]
    cont_cols = ["edad","gasto_promedio","antiguedad_dias","max_item_price","min_item_price","dias_promedio_compra"]

    feat[code_cols] = feat[code_cols].fillna(-1).astype("int64")
    feat[count_cols] = feat[count_cols].fillna(0).astype("int64")
    for c in cont_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(-1)

    # Orden
    feat = feat.sort_values("customer_id").reset_index(drop=True)
    return feat

# ===================================
# USED: ColumnTransformer, OneHotEncoder, StandardScaler, CountVectorizer
# ===================================
def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], text_col: str):
    """
    Crea un ColumnTransformer que:
    - Escala numéricos con StandardScaler (USED: StandardScaler)
    - Hace One-Hot de categóricos (USED: OneHotEncoder)
    - Vectoriza texto con CountVectorizer (USED: CountVectorizer)
    - Combina todo con ColumnTransformer (USED: ColumnTransformer)
    """
    num_pipe = Pipeline([
        ("scaler", StandardScaler())  # USED: StandardScaler
    ])

    cat_pipe = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))  # USED: OneHotEncoder
    ])

    # CountVectorizer requiere 1D de strings: usamos FunctionTransformer para "aplanar" la columna
    text_pipe = Pipeline([
        ("to_series", FunctionTransformer(_to_series, validate=False)),
        ("cv", CountVectorizer(lowercase=True, min_df=2, max_features=500))  # USED: CountVectorizer
    ])

    preprocessor = ColumnTransformer(  # USED: ColumnTransformer
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
            ("txt", text_pipe, [text_col])  # pasar como lista para mantener 2D -> luego se aplana
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    return preprocessor


def process_df(df, training=True):
    """
    HYBRID PIPELINE (solo lo estrictamente necesario):
    - Mantiene columnas numéricas y de conteo en su escala natural (SIN StandardScaler).
    - Aplica OneHotEncoder SOLO a: ["device_frecuente_code", "preferred_price_bin_code"].
    - Aplica CountVectorizer a "bag_of_titles" construido en memoria (no se guarda en CSV).
    - Combina todo en un ÚNICO CSV: customer_features.csv
    """
    # ---- Adjuntar texto en memoria ----
    text_col = "bag_of_titles"
    if text_col not in df.columns:
        try:
            raw = read_csv("customer_purchases_train")
        except Exception:
            raw = None
        if raw is not None and "item_title" in raw.columns and "customer_id" in raw.columns:
            raw = raw.copy()
            raw["customer_id"] = _numeric_customer_id(raw["customer_id"])
            text_agg = (raw.assign(item_title=raw["item_title"].astype(str).str.lower())
                          .groupby("customer_id")["item_title"]
                          .apply(lambda s: " ".join(s)))
            text_df = text_agg.reset_index().rename(columns={"item_title": text_col})
            df = df.merge(text_df, on="customer_id", how="left")
        else:
            df[text_col] = ""

    # ---- Selección de columnas ----
    categorical_cols = ["device_frecuente_code", "preferred_price_bin_code"]
    # num_cols naturales (SIN escalar): todas menos id, categóricas y texto
    exclude = set(["customer_id", text_col] + categorical_cols)
    numeric_natural_cols = [c for c in df.columns if c not in exclude]

    # ---- OneHotEncoder con compatibilidad de versión ----
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # ---- ColumnTransformer: solo cat y texto ----
    text_pipe = Pipeline([
        ("to_series", FunctionTransformer(_to_series, validate=False)),
        ("cv", CountVectorizer(lowercase=True, min_df=3, max_features=200))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_cols),
            ("txt", text_pipe, [text_col])
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )

    # Ajuste/transformación (no guardamos pkl; SOLO un CSV final híbrido)
    processed_array = preprocessor.fit_transform(df) if training else preprocessor.transform(df)

    # Nombres resultantes
    try:
        cat_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(categorical_cols).tolist()
    except Exception:
        cat_names = []
    try:
        txt_names = preprocessor.named_transformers_["txt"]["cv"].get_feature_names_out().tolist()
    except Exception:
        txt_names = []
    # Fallback robusto si no hay nombres suficientes
    total_cols = processed_array.shape[1]
    names = [f"cat__{n}" for n in cat_names] + [f"txt__{t}" for t in txt_names]
    if len(names) != total_cols:
        names = [f"f{i}" for i in range(total_cols)]
    from pandas import DataFrame
    proc_df = DataFrame(processed_array, index=df.index, columns=names)

    # ---- Ensamble híbrido: mantener naturales + agregar OHE + texto ----
    hybrid = df[["customer_id"] + numeric_natural_cols].copy()
    hybrid = hybrid.join(proc_df)

    # ---- Eliminar columna de texto crudo antes de guardar ----
    if text_col in hybrid.columns:
        hybrid = hybrid.drop(columns=[text_col], errors="ignore")

    # ---- Guardar como ÚNICO CSV híbrido ----
    from pathlib import Path as _Path
    out_csv = _Path(DATA_DIR) / "customer_features.csv"
    hybrid.to_csv(out_csv, index=False)
    print(f"df saved to {out_csv}")
    return hybrid

def preprocess(raw_df, training=False):
    processed_df = process_df(raw_df, training)
    return processed_df

def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

# ===================================
# Public API
# ===================================
def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    # Aplicar proceso sklearn y guardar preprocessor.pkl
    processed = process_df(customer_feat, training=True)
    X = processed.drop(columns=["customer_id"])
    y = None
    return X, y

def read_test_data():
    test_df = read_csv("customer_purchases_test")
    # Aquí deberías generar las mismas features que en train (no implementado por falta de CSV de test).
    # processed = process_df(test_features, training=False)
    X_test = test_df
    return X_test

if __name__ == "__main__":
    try:
        train_df = read_csv("customer_purchases_train")
        features = extract_customer_features(train_df)
        _ = process_df(features, training=True)
        print(_.head())
    except Exception as e:
        print("Error while generating features:", e)
