import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import ColumnTransformer
import joblib

import os
from pathlib import Path
from datetime import datetime
from negative_generation import (
    gen_all_negatives,
    gen_random_negatives,
)


DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


def save_df(df, filename: str):
    # Guardar
    save_path = os.path.join(DATA_DIR, filename)
    save_path = os.path.abspath(save_path)
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")


def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def extract_customer_features(df):
    # ['purchase_id',  'item_id', 'item_title',
    #    'item_category', 'item_price', 'item_img_filename', 'item_avg_rating',
    #    'item_num_ratings', 'item_release_date', 'purchase_timestamp']
    train_df = df.copy()
    customer_columns = [
        "customer_id",
        "customer_date_of_birth",
        "customer_gender",
        "customer_signup_date",
    ]
    today = datetime.strptime("2025-21-09", "%Y-%d-%m")
    dates = ["customer_date_of_birth", "customer_signup_date", "item_release_date", "purchase_timestamp"]
    for col in dates:
        train_df[col] = pd.to_datetime(train_df[col])

    group = train_df.groupby("customer_id")

    # -------- Tratamientos de fechas --------#
    age = (today - train_df["customer_date_of_birth"]).dt.days // 365
    train_df["customer_age_years"] = age.astype(int)
    customer_ages = group["customer_age_years"].first()

    tenure = (today - train_df["customer_signup_date"]).dt.days // 365
    train_df["customer_tenure_year"] = tenure.astype(int)
    customer_tenure = group["customer_tenure_year"].first()

    # ------------ Features agregados ---------#
    customer_price_stats = group["item_price"].agg(["mean", "std"])
    customer_price_stats.rename(
        columns={"mean": "item_avg_price", "std": "item_std_price"}, inplace=True
    )
    most_purchased_category = group["item_category"].agg(lambda x: x.mode()[0])

    # ========== NUEVOS FEATURES CON PREFIJO CUSTOMER ==========
    # 📊 COMPORTAMIENTO DE COMPRA
    customer_purchase_frequency = group.size()
    customer_recency_days = (today - group["purchase_timestamp"].max()).dt.days
    customer_total_spent = group["item_price"].sum()
    customer_avg_spent = group["item_price"].mean()
    customer_std_spent = group["item_price"].std().fillna(0)
    customer_min_spent = group["item_price"].min()
    customer_max_spent = group["item_price"].max()
    
    # 🎯 PREFERENCIAS DE PRODUCTO
    customer_category_diversity = group["item_category"].nunique()
    customer_preferred_price_range = group["item_price"].mean()
    
    # Extraer tipo de producto del título
    train_df["item_collection_type"] = train_df["item_title"].str.extract(r'(Premium|Exclusive|Modern|Elegant|Stylish)')
    customer_preferred_collection = group["item_collection_type"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown"
    )
    
    # ⏰ FEATURES TEMPORALES
    customer_tenure_days = (today - group["customer_signup_date"].first()).dt.days
    
    # Tiempo entre compras
    customer_avg_days_between_purchases = group["purchase_timestamp"].apply(
        lambda x: x.sort_values().diff().dt.days.mean() if len(x) > 1 else 0
    ).fillna(0)
    
    # Velocidad de adopción
    customer_avg_days_from_release = (
        (group["purchase_timestamp"].max() - group["item_release_date"].max()).dt.days
    )
    customer_prefers_new_items = (customer_avg_days_from_release < 30).astype(int)
    
    # 👀 COMPORTAMIENTO DE NAVEGACIÓN
    customer_avg_views = group["customer_item_views"].mean()
    customer_max_views = group["customer_item_views"].max()
    customer_views_to_purchase_ratio = group["customer_item_views"].sum() / customer_purchase_frequency
    
    # ⭐ RATING Y SATISFACCIÓN
    customer_rating_frequency = group["purchase_item_rating"].count() / customer_purchase_frequency
    customer_avg_rating_given = group["purchase_item_rating"].mean().fillna(0)
    customer_rating_behavior_vs_item = (
        group["purchase_item_rating"].mean() - group["item_avg_rating"].mean()
    ).fillna(0)
    
    # 📱 FEATURES TÉCNICAS
    customer_preferred_device = group["purchase_device"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else "unknown"
    )
    # CORRECCIÓN: Contar ocurrencias de "mobile" por grupo
    customer_mobile_ratio = group["purchase_device"].apply(lambda x: (x == "mobile").sum()) / customer_purchase_frequency

    # --------- Creando el dataframe ---------#
    customer_feat = pd.DataFrame(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_ages,
            "customer_tenure_years": customer_tenure,
            "customer_prefered_cat": most_purchased_category,
            
            # 📊 NUEVOS FEATURES DE COMPORTAMIENTO DE COMPRA
            "customer_purchase_frequency": customer_purchase_frequency,
            "customer_recency_days": customer_recency_days,
            "customer_total_spent": customer_total_spent,
            "customer_avg_spent": customer_avg_spent,
            "customer_std_spent": customer_std_spent,
            "customer_min_spent": customer_min_spent,
            "customer_max_spent": customer_max_spent,
            
            # 🎯 NUEVOS FEATURES DE PREFERENCIAS DE PRODUCTO
            "customer_category_diversity": customer_category_diversity,
            "customer_preferred_price_range": customer_preferred_price_range,
            "customer_preferred_collection": customer_preferred_collection,
            
            # ⏰ NUEVOS FEATURES TEMPORALES
            "customer_tenure_days": customer_tenure_days,
            "customer_avg_days_between_purchases": customer_avg_days_between_purchases,
            "customer_avg_days_from_release": customer_avg_days_from_release,
            "customer_prefers_new_items": customer_prefers_new_items,
            
            # 👀 NUEVOS FEATURES DE NAVEGACIÓN
            "customer_avg_views": customer_avg_views,
            "customer_max_views": customer_max_views,
            "customer_views_to_purchase_ratio": customer_views_to_purchase_ratio,
            
            # ⭐ NUEVOS FEATURES DE RATING
            "customer_rating_frequency": customer_rating_frequency,
            "customer_avg_rating_given": customer_avg_rating_given,
            "customer_rating_behavior_vs_item": customer_rating_behavior_vs_item,
            
            # 📱 NUEVOS FEATURES TÉCNICAS
            "customer_preferred_device": customer_preferred_device,
            "customer_mobile_ratio": customer_mobile_ratio,
        }
    ).reset_index(drop=True)

    save_df(customer_feat, "customer_features.csv")
    return customer_feat


def build_processor(
    df, numerical_features, categorical_features, free_text_features, training=True
):

    savepath = Path(DATA_DIR) / "preprocessor.pkl"
    if training:
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
        free_text_transformers = []
        for col in free_text_features:
            free_text_transformers.append(
                (
                    col,
                    CountVectorizer(),  # como quieren procesar esta columna?
                    col,
                )
            )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
                *free_text_transformers,
            ],
            remainder="passthrough",  # Mantener las demas sin tocar
        )
        df = df.drop(columns=["label"], errors="ignore")
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    # Numeric
    num_cols = numerical_features

    # Categorical
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )

    # Free-text
    bow_cols = []
    for col in free_text_features:
        vectorizer = preprocessor.named_transformers_[col]
        bow_cols.extend([f"{col}_bow_{t}" for t in vectorizer.get_feature_names_out()])

    # Passthrough
    other_cols = [
        c
        for c in df.columns
        if c not in numerical_features + categorical_features + free_text_features
    ]

    final_cols = list(num_cols) + list(cat_cols) + bow_cols + other_cols

    processed_df = pd.DataFrame(processed_array, columns=final_cols)
    return processed_df


def preprocess(raw_df, training=False):
    """
    Agrega tu procesamiento de datos, considera si necesitas guardar valores de entrenamiento.
    Utiliza la bandera para distinguir entre preprocesamiento de entrenamiento y validación/prueba
    """
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
    ]

    # normalization
    numerical_feat = [
        "item_price",
        "customer_age_years",
        "customer_tenure_years",
    ]

    # One hot
    raw_df["customer_cat_is_prefered"] = (
        raw_df["item_category"] == raw_df["customer_prefered_cat"]
    )
    categorical_features = ["customer_gender", "item_category", "item_img_filename"]

    # Texto
    free_text_features = ["item_title"]

    # Datetime
    datetimecols = []
    for col in datetimecols:
        raw_df[col] = pd.to_datetime(raw_df[col])

    # Ciclicas

    # ColumnTransformer
    processed_df = build_processor(
        raw_df,
        numerical_feat,
        categorical_features,
        free_text_features,
        training=training,
    )

    # Borrar columnas que no sirvan
    processed_df = processed_df.drop(columns=dropcols)
    return processed_df


def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)

    # -------------- Agregar negativos ------------------ #
    # Generar negativos
    train_df_neg = gen_random_negatives(train_df, n_per_positive=1)
    train_df_neg = train_df_neg.drop_duplicates(subset=["customer_id", "item_id"])

    # Agregar Features del cliente
    train_df_cust = pd.merge(train_df, customer_feat, on="customer_id", how="left")

    processed_pos = preprocess(train_df_cust, training=True)
    processed_pos["label"] = 1

    # Obtener todas las columnas
    all_columns = processed_pos.columns

    # Separar los features exclusivos de los items
    item_feat = [col for col in all_columns if "item" in col]
    unique_items = processed_pos[item_feat].drop_duplicates(
        subset=[
            "item_id",
        ]
    )

    # Separar los features exclusivos de los clientes
    customer_feat = [col for col in all_columns if "customer" in col]
    unique_customers = processed_pos[customer_feat].drop_duplicates(
        subset=["customer_id"]
    )

    # Agregar los features de los items a los negativos
    processed_neg = pd.merge(
        train_df_neg,
        unique_items,
        on=["item_id"],
        how="left",
    )

    # Agregar los features de los usuarios a los negativos
    processed_neg = pd.merge(
        processed_neg,
        unique_customers,
        on=["customer_id"],
        how="left",
    )

    # Agregar etiqueta a los negativos
    processed_neg["label"] = 0

    # Combinar negativos con positivos para tener el dataset completo
    processed_full = (
        pd.concat([processed_pos, processed_neg], axis=0)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    # Transformar a tipo numero
    shuffled = df_to_numeric(processed_full)
    y = shuffled["label"]

    # Eliminar columnas que no sirven
    X = shuffled.drop(columns=["label", "customer_id", "item_id"])
    return X, y


def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")

    # agregar features derivados del cliente al dataset
    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")

    # Procesamiento de datos
    processed = preprocess(merged, training=False)

    # Si se requiere
    dropcols = []
    processed = processed.drop(columns=dropcols)

    return df_to_numeric(processed)


if __name__ == "__main__":
    X_train, y_train = read_train_data()
    print(X_train.info())
    test_df = read_csv("customer_purchases_test")

    X_test = read_test_data()
    print(test_df.columns)
    print("hola")