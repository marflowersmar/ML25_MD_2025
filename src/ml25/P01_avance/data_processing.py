# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
import joblib
import os
from pathlib import Path
from datetime import datetime

# Negativos
from negative_generation import gen_random_negatives

# -----------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21)

CURRENT_FILE = Path(__file__).resolve()
# src/ml25/P01_avance -> subimos a src/ml25 -> datasets/customer_purchases
DEFAULT_DATA_DIR = (CURRENT_FILE.parent.parent / "datasets" / "customer_purchases").resolve()
DATA_DIR = Path(os.getenv("CUSTOMER_PURCHASES_DIR", str(DEFAULT_DATA_DIR))).resolve()

# -----------------------------------------
# AUXILIARES
# -----------------------------------------
def _ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la ruta: {path}\n"
            f"Verifica que el dataset esté en {DATA_DIR}\n"
            f"o define CUSTOMER_PURCHASES_DIR como variable de entorno."
        )

def read_csv(filename: str):
    file = DATA_DIR / f"{filename}.csv"
    _ensure_exists(file)
    return pd.read_csv(file)

def save_df(df: pd.DataFrame, filename: str):
    out = DATA_DIR / (filename if filename.endswith(".csv") else f"{filename}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"DataFrame guardado en: {out}")

def df_to_numeric(df: pd.DataFrame):
    """Convierte a numérico cuando aplique, evitando tocar columnas que ya son numéricas o sparse."""
    data = df.copy()
    for c in data.columns:
        s = data[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_sparse(s.dtype):
            data[c] = s.astype("float64")
            continue
        try:
            data[c] = pd.to_numeric(s, errors="coerce").fillna(0)
        except Exception as e:
            print(f"Advertencia: {c} no convertible: {e}")
            continue
    return data

def _safe_text_col(df: pd.DataFrame, name: str) -> pd.Series:
    """Serie de texto lower() sin NaN. Si no existe, serie vacía."""
    if name in df.columns:
        return df[name].fillna("").astype(str).str.lower()
    return pd.Series([""] * len(df), index=df.index, dtype="object")

def _factorize_code(series: pd.Series, unknown_value: str) -> np.ndarray:
    s = series.fillna(unknown_value).astype(str)
    codes, _ = pd.factorize(s, sort=True)
    return codes.astype(int)

def _price_bin(x: float) -> str:
    if pd.isna(x): return "p_nan"
    if x < 200:   return "p_0_200"
    if x < 500:   return "p_200_500"
    if x < 1000:  return "p_500_1000"
    return "p_1000_plus"

def _map_img_to_color(filename: str) -> str:
    """
    Mapea item_img_filename a color canónico.
    imgb->blue, imgbl->black, imgg->green, imgo->orange, imgp->pink, imgr->red, imgw->white, imgy->yellow
    """
    if not isinstance(filename, str): return ""
    f = filename.lower()
    if "imgbl" in f: return "black"
    if "imgb"  in f: return "blue"
    if "imgg"  in f: return "green"
    if "imgo"  in f: return "orange"
    if "imgp"  in f: return "pink"
    if "imgr"  in f: return "red"
    if "imgw"  in f: return "white"
    if "imgy"  in f: return "yellow"
    return ""

# -----------------------------------------
# FEATURE ENGINEERING DE CLIENTE
# -----------------------------------------
def extract_customer_features(df: pd.DataFrame):
    """
    Construye un dataset por cliente con señales previas y nuevas.
    """
    train_df = df.copy()

    # Fechas a datetime
    for col in [
        "customer_date_of_birth",
        "customer_signup_date",
        "item_release_date",
        "purchase_timestamp",
    ]:
        if col in train_df.columns:
            train_df[col] = pd.to_datetime(train_df[col], errors="coerce")

    # Normalizaciones seguras de texto
    train_df["item_category"] = _safe_text_col(train_df, "item_category")
    train_df["item_title"] = _safe_text_col(train_df, "item_title")
    train_df["item_img_filename"] = _safe_text_col(train_df, "item_img_filename")
    train_df["customer_gender"] = _safe_text_col(train_df, "customer_gender")
    train_df["purchase_device"] = _safe_text_col(train_df, "purchase_device")

    today = DATA_COLLECTED_AT
    group = train_df.groupby("customer_id", dropna=False)

    # Señales históricas
    customer_purchase_frequency = group.size()
    customer_recency_days = (today - group["purchase_timestamp"].max()).dt.days
    customer_total_spent = group["item_price"].sum()
    customer_avg_spent = group["item_price"].mean()
    customer_std_spent = group["item_price"].std().fillna(0)
    customer_min_spent = group["item_price"].min()
    customer_max_spent = group["item_price"].max()

    customer_prefered_cat = group["item_category"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown")
    customer_category_diversity = group["item_category"].nunique()

    # Edad y antigüedad
    customer_age_years_series = ((today - group["customer_date_of_birth"].first()).dt.days // 365).astype(float)
    customer_tenure_years_series = ((today - group["customer_signup_date"].first()).dt.days // 365).astype(float)
    customer_tenure_days = (today - group["customer_signup_date"].first()).dt.days

    # Espaciado entre compras
    customer_avg_days_between_purchases = group["purchase_timestamp"].apply(
        lambda s: pd.to_datetime(s, errors="coerce").sort_values().diff().dt.days.mean() if len(s) > 1 else 0
    ).fillna(0)

    # Nuevos atributos
    customer_edad = ((today - group["customer_date_of_birth"].first()).dt.days // 365).astype(float)

    gender_mode = group["customer_gender"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown")
    customer_gender_code = _factorize_code(gender_mode, unknown_value="unknown")

    customer_total_compras = customer_purchase_frequency.astype(float)
    customer_gasto_promedio = group["item_price"].mean().astype(float)
    customer_antiguedad_dias = (today - group["customer_signup_date"].first()).dt.days.astype(float)

    cat_mode = customer_prefered_cat.reindex(customer_purchase_frequency.index).fillna("unknown")
    customer_categoria_frecuente_code = _factorize_code(cat_mode, unknown_value="unknown")

    dev_mode = group["purchase_device"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown_device")
    customer_device_frecuente_code = _factorize_code(dev_mode, unknown_value="unknown_device")

    price_mean_per_cust = group["item_price"].mean()
    price_bin_labels = price_mean_per_cust.apply(_price_bin)
    customer_preferred_price_bin_code = _factorize_code(price_bin_labels, unknown_value="p_nan")

    customer_max_item_price = group["item_price"].max().astype(float)
    customer_min_item_price = group["item_price"].min().astype(float)
    customer_dias_promedio_compra = customer_avg_days_between_purchases.astype(float)

    # Colores por filename
    train_df["_color_mapped"] = train_df["item_img_filename"].map(_map_img_to_color)
    color_list = ["black", "blue", "green", "orange", "pink", "red", "white", "yellow"]
    color_counts = {
        f"customer_color_{c}_count": group["_color_mapped"].apply(lambda s, c=c: (s == c).sum()).astype(int)
        for c in color_list
    }

    # Categorías de ropa
    cat_targets = ["blouse", "dress", "jacket", "jeans", "shirt", "shoes", "skirt", "slacks", "suit", "t-shirt"]
    cat_counts = {
        f"customer_cat_{c}_count": group["item_category"].apply(lambda s, c=c: (s == c).sum()).astype(int)
        for c in cat_targets
    }

    # Ensamble final por cliente
    base_index = customer_purchase_frequency.index
    customer_feat = pd.DataFrame(
        {
            "customer_id": base_index,
            "customer_purchase_frequency": customer_purchase_frequency.values,
            "customer_recency_days": customer_recency_days.values,
            "customer_total_spent": customer_total_spent.values,
            "customer_avg_spent": customer_avg_spent.values,
            "customer_std_spent": customer_std_spent.values,
            "customer_min_spent": customer_min_spent.values,
            "customer_max_spent": customer_max_spent.values,
            "customer_prefered_cat": cat_mode.values,
            "customer_category_diversity": customer_category_diversity.values,
            "customer_age_years": customer_age_years_series.reindex(base_index).values,
            "customer_tenure_years": customer_tenure_years_series.reindex(base_index).values,
            "customer_tenure_days": customer_tenure_days.values,
            "customer_avg_days_between_purchases": customer_avg_days_between_purchases.values,
            "customer_edad": customer_edad.reindex(base_index).values,
            "customer_gender_code": customer_gender_code,
            "customer_total_compras": customer_total_compras.values,
            "customer_gasto_promedio": customer_gasto_promedio.reindex(base_index).values,
            "customer_antiguedad_dias": customer_antiguedad_dias.reindex(base_index).values,
            "customer_categoria_frecuente_code": customer_categoria_frecuente_code,
            "customer_device_frecuente_code": customer_device_frecuente_code,
            "customer_preferred_price_bin_code": customer_preferred_price_bin_code,
            "customer_max_item_price": customer_max_spent.reindex(base_index).values if 'customer_max_spent' in locals() else customer_max_spent.reindex(base_index).values,
            "customer_min_item_price": customer_min_spent.reindex(base_index).values if 'customer_min_spent' in locals() else customer_min_spent.reindex(base_index).values,
            "customer_dias_promedio_compra": customer_dias_promedio_compra.reindex(base_index).values,
        }
    )

    # Agregar color counts
    for k, v in color_counts.items():
        customer_feat[k] = v.reindex(base_index).values

    # Agregar categoría counts
    for k, v in cat_counts.items():
        customer_feat[k] = v.reindex(base_index).values

    save_df(customer_feat, "customer_features.csv")
    return customer_feat

# -----------------------------------------
# PREPROCESAMIENTO
# -----------------------------------------
def build_processor(df, numerical_features, categorical_features, free_text_features, training=True):
    savepath = DATA_DIR / "preprocessor.pkl"

    if training:
        # Normaliza tipos categóricos a string antes de fit
        df_fit = df.copy()
        for col in categorical_features:
            if col in df_fit.columns:
                df_fit[col] = df_fit[col].astype(str).fillna("unknown")

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        X_numcat = preprocessor.fit_transform(df_fit)
        joblib.dump(preprocessor, savepath)
        print(f"Preprocessor entrenado y guardado en: {savepath}")

        bow_frames = []
        for col in free_text_features:
            vectorizer = CountVectorizer(max_features=3000, min_df=3)
            bow = vectorizer.fit_transform(df[col].astype(str))
            bow_df = pd.DataFrame(
                bow.toarray(),
                columns=[f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()]
            )
            bow_frames.append(bow_df)
            joblib.dump(vectorizer, savepath.with_name(f"{col}_vectorizer.pkl"))

    else:
        preprocessor = joblib.load(savepath)

        # Alinear tipos de categóricas al dtype aprendido por el preprocesador
        df_inf = df.copy()
        cat_transformer = preprocessor.named_transformers_["cat"]
        fitted_categories = cat_transformer.categories_

        if len(fitted_categories) != len(categorical_features):
            raise RuntimeError(
                "Desalineación entre categorical_features y categorías aprendidas del preprocesador"
            )

        for i, col in enumerate(categorical_features):
            if col not in df_inf.columns:
                continue
            cats = fitted_categories[i]
            # Si categorías aprendidas son numéricas, castear a numérico; si son strings, castear a str
            if hasattr(cats, "dtype") and np.issubdtype(cats.dtype, np.number):
                df_inf[col] = pd.to_numeric(df_inf[col], errors="coerce").fillna(-1)
                try:
                    df_inf[col] = df_inf[col].astype(cats.dtype)
                except Exception:
                    df_inf[col] = pd.to_numeric(df_inf[col], errors="coerce").fillna(-1)
            else:
                df_inf[col] = df_inf[col].astype(str).fillna("unknown")

        X_numcat = preprocessor.transform(df_inf)

        bow_frames = []
        for col in free_text_features:
            vectorizer_path = savepath.with_name(f"{col}_vectorizer.pkl")
            vectorizer = joblib.load(vectorizer_path)
            bow = vectorizer.transform(df[col].astype(str))
            bow_df = pd.DataFrame(
                bow.toarray(),
                columns=[f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()]
            )
            bow_frames.append(bow_df)

    # ---------------------------------------------------------
    # Reconstrucción robusta de nombres de columnas
    # ---------------------------------------------------------
    num_names = list(numerical_features)
    try:
        num_tr = preprocessor.named_transformers_["num"]
        if hasattr(num_tr, "feature_names_in_"):
            num_names = list(num_tr.feature_names_in_)
    except Exception:
        pass

    enc = preprocessor.named_transformers_["cat"]
    try:
        if hasattr(enc, "feature_names_in_"):
            cat_cols = enc.get_feature_names_out(enc.feature_names_in_)
        else:
            cat_cols = enc.get_feature_names_out()
    except Exception:
        try:
            cat_cols = enc.get_feature_names_out()
        except Exception:
            # Fallback extremo
            cat_cols = [f"cat_{i}" for i in range(X_numcat.shape[1] - len(num_names))]

    numcat_df = pd.DataFrame(X_numcat, columns=list(num_names) + list(cat_cols))
    final_df = pd.concat([numcat_df] + bow_frames, axis=1)
    return final_df

def preprocess(raw_df, training=False):
    raw_df = raw_df.copy()

    # Limpieza robusta
    categorical_columns = ['item_category', 'customer_gender', 'customer_prefered_cat', 'customer_fav_category']
    for col in categorical_columns:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].fillna("unknown")

    text_columns = ['item_title']
    for col in text_columns:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].fillna("")

    # Nunca metas IDs al modelo
    for _id in ["purchase_id", "customer_id", "item_id", "label"]:
        if _id in raw_df.columns:
            raw_df = raw_df.drop(columns=[_id])

    for col, val in {"item_category": "unknown_category", "item_title": ""}.items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].fillna(val)

    # ---------- LISTAS DE FEATURES ----------
    numerical_features = [
        "item_price",
        "customer_age_years",
        "customer_tenure_years",
        "customer_std_spent",
        "customer_recency_days",
        "customer_total_spent",
        "customer_avg_spent",
        "customer_min_spent",
        "customer_max_spent",
        "customer_avg_days_between_purchases",
        "customer_edad",
        "customer_total_compras",
        "customer_gasto_promedio",
        "customer_antiguedad_dias",
        "customer_max_item_price",
        "customer_min_item_price",
        "customer_dias_promedio_compra",
        "customer_color_black_count",
        "customer_color_blue_count",
        "customer_color_green_count",
        "customer_color_orange_count",
        "customer_color_pink_count",
        "customer_color_red_count",
        "customer_color_white_count",
        "customer_color_yellow_count",
        "customer_cat_blouse_count",
        "customer_cat_dress_count",
        "customer_cat_jacket_count",
        "customer_cat_jeans_count",
        "customer_cat_shirt_count",
        "customer_cat_shoes_count",
        "customer_cat_skirt_count",
        "customer_cat_slacks_count",
        "customer_cat_suit_count",
        "customer_cat_t-shirt_count",
    ]

    categorical_features = []
    if "customer_gender" in raw_df.columns:
        categorical_features.append("customer_gender")
    categorical_features += [
        "item_category",
        "customer_prefered_cat" if "customer_prefered_cat" in raw_df.columns else "item_category",
        "customer_gender_code",
        "customer_categoria_frecuente_code",
        "customer_device_frecuente_code",
        "customer_preferred_price_bin_code",
    ]

    free_text_features = ["item_title"]

    # Asegurar existencia de columnas esperadas
    for col in numerical_features:
        if col not in raw_df.columns:
            raw_df[col] = 0
    for col in categorical_features:
        if col not in raw_df.columns:
            raw_df[col] = "missing"
    for col in free_text_features:
        if col not in raw_df.columns:
            raw_df[col] = ""

    # Relleno de NaN numéricos antes del scaler
    for col in numerical_features:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").fillna(0)

    processed_df = build_processor(
        raw_df, numerical_features, categorical_features, free_text_features, training=training
    )
    # processed_df ya es numérico
    return processed_df

# -----------------------------------------
# TABLA CANÓNICA PARA ENTRENAR
# -----------------------------------------
def build_training_table(n_per_positive=1, smart=True, random_state=42):
    """
    Devuelve X_raw, y para split -> preprocess(train/val).
    Usa customer_features.csv; si está incompleto o no existe, lo recalcula.
    """
    train_df = read_csv("customer_purchases_train")

    # Customer features
    need_cols = [
        "customer_edad", "customer_gender_code", "customer_total_compras",
        "customer_gasto_promedio", "customer_antiguedad_dias",
        "customer_categoria_frecuente_code", "customer_device_frecuente_code",
        "customer_preferred_price_bin_code", "customer_max_item_price",
        "customer_min_item_price", "customer_dias_promedio_compra",
        "customer_color_black_count", "customer_color_blue_count", "customer_color_green_count",
        "customer_color_orange_count", "customer_color_pink_count", "customer_color_red_count",
        "customer_color_white_count", "customer_color_yellow_count",
        "customer_cat_blouse_count", "customer_cat_dress_count", "customer_cat_jacket_count",
        "customer_cat_jeans_count", "customer_cat_shirt_count", "customer_cat_shoes_count",
        "customer_cat_skirt_count", "customer_cat_slacks_count", "customer_cat_suit_count",
        "customer_cat_t-shirt_count",
    ]
    try:
        customer_feat = read_csv("customer_features")
        if any(col not in customer_feat.columns for col in need_cols):
            raise FileNotFoundError("customer_features incompleto")
    except FileNotFoundError:
        customer_feat = extract_customer_features(train_df)

    # Positivos del histórico
    pos = train_df.merge(customer_feat, on="customer_id", how="left")
    pos["label"] = 1

    # Negativos base
    neg = gen_random_negatives(train_df, n_per_positive=n_per_positive, smart=smart, random_state=random_state)
    neg = neg.merge(customer_feat, on="customer_id", how="left")
    neg["label"] = 0

    # Enriquecer negativos con atributos del ítem para evitar NaN en preprocess
    item_tbl = train_df[["item_id", "item_price", "item_category", "item_title"]].drop_duplicates("item_id")
    neg = neg.merge(item_tbl, on="item_id", how="left")

    # Fallbacks seguros
    neg["item_category"] = neg["item_category"].fillna("unknown_category").astype(str)
    neg["item_title"] = neg["item_title"].fillna("").astype(str)
    neg["item_price"] = pd.to_numeric(neg["item_price"], errors="coerce").fillna(0.0)

    # Concat + shuffle
    full = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=random_state)

    # X_raw y y sin IDs
    y = full["label"].copy()
    X_raw = full.drop(columns=["label", "customer_id", "item_id", "purchase_id"], errors="ignore")
    return X_raw, y

# -----------------------------------------
# DATOS DE TEST
# -----------------------------------------
def read_test_data():
    test_df = read_csv("customer_purchases_test")

    # Asegurar customer_features con los nuevos campos
    try:
        customer_feat = read_csv("customer_features")
        check_cols = ["customer_edad", "customer_gender_code", "customer_total_compras"]
        if any(c not in customer_feat.columns for c in check_cols):
            raise FileNotFoundError
    except FileNotFoundError:
        train_df = read_csv("customer_purchases_train")
        customer_feat = extract_customer_features(train_df)

    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")
    processed = preprocess(merged, training=False)
    return processed

# -----------------------------------------
# PRUEBA LOCAL
# -----------------------------------------
if __name__ == "__main__":
    _ensure_exists(DATA_DIR)
    print(f"Usando DATA_DIR = {DATA_DIR}")

    X_raw, y = build_training_table(n_per_positive=1, smart=True)
    from sklearn.model_selection import train_test_split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)
    print("X_train:", X_tr.shape, " | X_val:", X_va.shape)
