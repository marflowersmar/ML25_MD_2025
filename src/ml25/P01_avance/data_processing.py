import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
import joblib
import os
from pathlib import Path
from datetime import datetime

from negative_generation import gen_random_negatives

# -----------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21)
DATA_DIR = Path(r"C:\Users\fomm0\OneDrive\Documents\ICE V\APRENDIZAJE DE MAQUINA\ML25_MD_2025\src\ml25\datasets\customer_purchases")

# -----------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------
def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    return pd.read_csv(file)

def save_df(df, filename: str):
    save_path = os.path.join(DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"DataFrame guardado en: {save_path}")

def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0)
    return data

# -----------------------------------------
# FEATURE ENGINEERING DE CLIENTE (EXTENDIDO)
# -----------------------------------------
def extract_customer_features(df):
    train_df = df.copy()
    for col in [
        "customer_date_of_birth",
        "customer_signup_date",
        "item_release_date",
        "purchase_timestamp",
    ]:
        train_df[col] = pd.to_datetime(train_df[col], errors="coerce")

    today = DATA_COLLECTED_AT
    group = train_df.groupby("customer_id")

    # --- COMPORTAMIENTO DE COMPRA
    customer_purchase_frequency = group.size()
    customer_recency_days = (today - group["purchase_timestamp"].max()).dt.days
    customer_total_spent = group["item_price"].sum()
    customer_avg_spent = group["item_price"].mean()
    customer_std_spent = group["item_price"].std().fillna(0)
    customer_min_spent = group["item_price"].min()
    customer_max_spent = group["item_price"].max()

    # --- PREFERENCIAS
    customer_prefered_cat = group["item_category"].agg(lambda x: x.mode()[0] if len(x.mode()) else "Unknown")
    customer_category_diversity = group["item_category"].nunique()
    train_df["item_collection_type"] = train_df["item_title"].str.extract(r'(Premium|Exclusive|Modern|Elegant|Stylish)', expand=False)
    customer_preferred_collection = group["item_collection_type"].agg(lambda x: x.mode()[0] if len(x.mode()) else "Unknown")
    customer_avg_days_from_release = (group["purchase_timestamp"].mean() - group["item_release_date"].mean()).dt.days
    customer_prefers_new_items = (customer_avg_days_from_release < 30).astype(int)
    train_df["item_img_type"] = train_df["item_img_filename"].str.extract(r'(imgr|imgp|imgbl|imgw)', expand=False)
    customer_fav_img_type = group["item_img_type"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown")

    # --- TEMPORALES
    customer_age_years = ((today - train_df["customer_date_of_birth"]).dt.days // 365).astype(float)
    customer_tenure_years = ((today - train_df["customer_signup_date"]).dt.days // 365).astype(float)
    customer_tenure_days = (today - group["customer_signup_date"].first()).dt.days
    train_df["purchase_month"] = train_df["purchase_timestamp"].dt.month
    customer_fav_month = group["purchase_month"].agg(lambda x: x.mode()[0] if len(x.mode()) else 0)
    customer_avg_days_between_purchases = group["purchase_timestamp"].apply(
        lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else 0
    ).fillna(0)
    customer_adoption_speed = (group["purchase_timestamp"].mean() - group["item_release_date"].mean()).dt.days

    # --- NAVEGACIÓN
    customer_avg_views = group["customer_item_views"].mean()
    customer_max_views = group["customer_item_views"].max()
    customer_views_to_purchase_ratio = (group["customer_item_views"].sum() / customer_purchase_frequency).fillna(0)

    # --- RATING
    customer_rating_frequency = (group["purchase_item_rating"].count() / customer_purchase_frequency).fillna(0)
    customer_avg_rating_given = group["purchase_item_rating"].mean().fillna(0)
    customer_rating_behavior_vs_item = (
        group["purchase_item_rating"].mean() - group["item_avg_rating"].mean()
    ).fillna(0)
    customer_prefers_popular_items = group["item_num_ratings"].mean().fillna(0)

    # --- TÉCNICAS + ESTABLES (MODO POR CLIENTE)
    customer_preferred_device = group["purchase_device"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown")
    customer_mobile_ratio = group["purchase_device"].apply(lambda x: (x == "mobile").sum() / len(x))

    # Estables para el modelo (evitan NaN sistemáticos)
    customer_gender_mode = group["customer_gender"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown_gender")
    customer_device_mode = group["purchase_device"].agg(lambda x: x.mode()[0] if len(x.mode()) else "unknown_device")

    customer_feat = pd.DataFrame({
        "customer_id": group["customer_id"].first(),
        # Comportamiento
        "customer_purchase_frequency": customer_purchase_frequency,
        "customer_recency_days": customer_recency_days,
        "customer_total_spent": customer_total_spent,
        "customer_avg_spent": customer_avg_spent,
        "customer_std_spent": customer_std_spent,
        "customer_min_spent": customer_min_spent,
        "customer_max_spent": customer_max_spent,
        # Preferencias
        "customer_prefered_cat": customer_prefered_cat,
        "customer_category_diversity": customer_category_diversity,
        "customer_preferred_collection": customer_preferred_collection,
        "customer_prefers_new_items": customer_prefers_new_items,
        "customer_fav_img_type": customer_fav_img_type,
        # Temporales
        "customer_age_years": customer_age_years.groupby(train_df["customer_id"]).first(),
        "customer_tenure_years": customer_tenure_years.groupby(train_df["customer_id"]).first(),
        "customer_tenure_days": customer_tenure_days,
        "customer_fav_month": customer_fav_month,
        "customer_avg_days_between_purchases": customer_avg_days_between_purchases,
        "customer_adoption_speed": customer_adoption_speed,
        # Navegación
        "customer_avg_views": customer_avg_views,
        "customer_max_views": customer_max_views,
        "customer_views_to_purchase_ratio": customer_views_to_purchase_ratio,
        # Rating
        "customer_rating_frequency": customer_rating_frequency,
        "customer_avg_rating_given": customer_avg_rating_given,
        "customer_rating_behavior_vs_item": customer_rating_behavior_vs_item,
        "customer_prefers_popular_items": customer_prefers_popular_items,
        # Técnicas (históricas)
        "customer_preferred_device": customer_preferred_device,
        "customer_mobile_ratio": customer_mobile_ratio,
        # Estables para el modelo
        "customer_gender_mode": customer_gender_mode,
        "customer_device_mode": customer_device_mode,
    }).reset_index(drop=True)

    save_df(customer_feat, "customer_features.csv")
    return customer_feat

# -----------------------------------------
# PREPROCESAMIENTO GENERAL
# -----------------------------------------
def build_processor(df, numerical_features, categorical_features, free_text_features, training=True):
    savepath = Path(DATA_DIR) / "preprocessor.pkl"

    if training:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        X_numcat = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
        print(f"Preprocessor entrenado y guardado en: {savepath}")

        bow_frames = []
        for col in free_text_features:
            vectorizer = CountVectorizer()
            bow = vectorizer.fit_transform(df[col].astype(str))
            bow_df = pd.DataFrame(
                bow.toarray(),
                columns=[f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()]
            )
            bow_frames.append(bow_df)
            joblib.dump(vectorizer, savepath.with_name(f"{col}_vectorizer.pkl"))
    else:
        preprocessor = joblib.load(savepath)
        X_numcat = preprocessor.transform(df)

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

    num_cols = numerical_features
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    numcat_df = pd.DataFrame(X_numcat, columns=list(num_cols) + list(cat_cols))
    final_df = pd.concat([numcat_df] + bow_frames, axis=1)
    return final_df

def preprocess(raw_df, training=False):
    raw_df = raw_df.copy()

    # Señal derivada opcional
    raw_df["customer_cat_is_prefered"] = (
        raw_df.get("item_category") == raw_df.get("customer_prefered_cat", "")
    )

    # Selección tolerante de columnas (usa *_mode si existe; si no, base)
    gender_col = "customer_gender_mode" if "customer_gender_mode" in raw_df.columns else (
        "customer_gender" if "customer_gender" in raw_df.columns else None
    )
    device_col = "customer_device_mode" if "customer_device_mode" in raw_df.columns else (
        "customer_preferred_device" if "customer_preferred_device" in raw_df.columns else None
    )

    # Limpieza de NaN (evita categorías que aparezcan solo en negativos)
    fill_values = {
        "item_category": "unknown_category",
        "item_title": "",
    }
    if gender_col:
        fill_values[gender_col] = "unknown_gender"
    if device_col:
        fill_values[device_col] = "unknown_device"

    for col, val in fill_values.items():
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].fillna(val)

    # Conjunto de features
    numerical_features = ["item_price", "customer_age_years", "customer_tenure_years"]
    categorical_features = []
    if gender_col:
        categorical_features.append(gender_col)
    categorical_features.append("item_category")
    if device_col:
        categorical_features.append(device_col)

    free_text_features = ["item_title"]

    processed_df = build_processor(
        raw_df, numerical_features, categorical_features, free_text_features, training=training
    )
    processed_df = df_to_numeric(processed_df)
    return processed_df

# -----------------------------------------
# NUEVO: TABLA CANÓNICA PARA ENTRENAR
# -----------------------------------------
def build_training_table(n_per_positive=1, smart=True, random_state=42):
    """
    Devuelve X_raw, y listos para split → preprocess(train/val).
    - Enriquecer negativos con atributos de item.
    - Aportar atributos ESTABLES de cliente a pos/neg.
    - Regenerar customer_features si faltan columnas nuevas.
    """
    # Base
    train_df = read_csv("customer_purchases_train")

    # Customer features (si no existe o le faltan columnas, recalcular)
    try:
        customer_feat = read_csv("customer_features")
    except FileNotFoundError:
        customer_feat = extract_customer_features(train_df)

    required_cols = [
        "customer_gender_mode",
        "customer_device_mode",
        "customer_age_years",
        "customer_tenure_years",
    ]
    if any(col not in customer_feat.columns for col in required_cols):
        customer_feat = extract_customer_features(train_df)

    # Positivos (compras reales) + join de features cliente
    pos = train_df.merge(customer_feat, on="customer_id", how="left")
    pos["label"] = 1

    # Negativos enriquecidos (ya incluyen item_title, item_category, item_price, item_img_filename)
    neg = gen_random_negatives(train_df, n_per_positive=n_per_positive, smart=smart, random_state=random_state)
    neg = neg.merge(customer_feat, on="customer_id", how="left")
    neg["label"] = 0

    # Concat + shuffle
    full = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=random_state)

    # X_raw y y (sin ID)
    y = full["label"].copy()
    X_raw = full.drop(columns=["label", "customer_id", "item_id"], errors="ignore")
    return X_raw, y

# -----------------------------------------
# DATOS DE TEST (para inferencia de producción)
# -----------------------------------------
def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")
    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")
    processed = preprocess(merged, training=False)
    return df_to_numeric(processed)

# -----------------------------------------
# PRUEBA LOCAL
# -----------------------------------------
if __name__ == "__main__":
    # Prueba del pipeline canónico
    X_raw, y = build_training_table(n_per_positive=1, smart=True)
    from sklearn.model_selection import train_test_split
    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=42)
    X_tr = preprocess(X_tr_raw, training=True)
    X_va = preprocess(X_va_raw, training=False)
    print("X_train:", X_tr.shape, " X_val:", X_va.shape)
