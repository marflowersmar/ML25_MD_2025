import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.compose import ColumnTransformer
import joblib

import os
from pathlib import Path
from datetime import datetime
from ml25.P01_customer_purchases.boilerplate.negative_generation import (
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
    dates = ["customer_date_of_birth", "customer_signup_date"]
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

    # --------- Creando el dataframe ---------#
    customer_feat = pd.DataFrame(
        {
            "customer_id": group["customer_id"].first(),
            "customer_age_years": customer_ages,
            "customer_tenure_years": customer_tenure,
            "customer_prefered_cat": most_purchased_category,
        }
    ).reset_index(drop=True)

    save_df(customer_feat, "customer_features.csv")
    return customer_feat


def build_processor(
    df, numerical_features, categorical_features, free_text_features, training=True
):
    """
    Fits or loads a ColumnTransformer preprocessor and returns a processed DataFrame.

    Parameters:
    - df: input DataFrame
    - numerical_features: list of numeric columns to scale
    - categorical_features: list of categorical columns to one-hot encode
    - training: if True, fit and save preprocessor; if False, load preprocessor
    - data_dir: directory to save/load preprocessor
    - preprocessor_name: filename for preprocessor

    Returns:
    - processed_df: DataFrame with transformed numeric & categorical columns + passthrough columns
    - preprocessor: fitted ColumnTransformer
    """
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
