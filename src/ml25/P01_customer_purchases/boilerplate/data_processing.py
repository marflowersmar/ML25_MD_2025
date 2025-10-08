import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../../datasets/customer_purchases/"

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df

def save_df(df, filename: str):
    save_path = os.path.join(DATA_DIR, "customer_features.csv")
    df.to_csv(save_path, index=False)
    print(f"df saved to {save_path}")

def extract_customer_features(train_df):
    # Atributos derivados antes del agrupamiento
    train_df["customer_date_of_birth"] = pd.to_datetime(train_df["customer_date_of_birth"], errors="coerce")
    train_df["edad"] = (DATA_COLLECTED_AT - train_df["customer_date_of_birth"].dt.date).dt.days // 365

    train_df["customer_signup_date"] = pd.to_datetime(train_df["customer_signup_date"], errors="coerce")
    train_df["antiguedad_dias"] = (DATA_COLLECTED_AT - train_df["customer_signup_date"].dt.date).dt.days

    train_df["customer_gender"] = train_df["customer_gender"].str.lower()
    train_df["gender_code"] = train_df["customer_gender"].map({"female": 0, "male": 1})
    train_df["gender_code"].fillna(-1, inplace=True)

    agg_price = train_df.groupby("customer_id")["item_price"].agg(["max", "min"]).reset_index()
    agg_price.rename(columns={"max": "max_item_price", "min": "min_item_price"}, inplace=True)
    train_df = train_df.merge(agg_price, on="customer_id", how="left")

    gasto = train_df.groupby("customer_id")["item_price"].mean().rename("gasto_promedio")
    total = train_df.groupby("customer_id")["item_id"].count().rename("total_compras")
    train_df = train_df.merge(gasto, on="customer_id", how="left")
    train_df = train_df.merge(total, on="customer_id", how="left")

    cat_frec = train_df.groupby("customer_id")["item_category"].agg(lambda x: x.value_counts().idxmax())
    dev_frec = train_df.groupby("customer_id")["purchase_device"].agg(lambda x: x.value_counts().idxmax())
    train_df = train_df.merge(cat_frec.rename("categoria_frecuente"), on="customer_id", how="left")
    train_df = train_df.merge(dev_frec.rename("device_frecuente"), on="customer_id", how="left")

    train_df["categoria_frecuente_code"] = train_df["categoria_frecuente"].astype("category").cat.codes
    train_df["device_frecuente_code"] = train_df["device_frecuente"].astype("category").cat.codes

    bins = [0, 500, 1000, 1500, 2000, 5000, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    train_df["preferred_price_bin_code"] = pd.cut(train_df["item_price"], bins=bins, labels=labels).astype(int)

    train_df["purchase_timestamp"] = pd.to_datetime(train_df["purchase_timestamp"], errors="coerce")
    intervalos = train_df.sort_values(["customer_id", "purchase_timestamp"]).groupby("customer_id")["purchase_timestamp"].diff().dt.days
    dias_promedio = intervalos.groupby(train_df["customer_id"]).mean().fillna(0)
    train_df = train_df.merge(dias_promedio.rename("dias_promedio_compra"), on="customer_id", how="left")

    customer_feat = train_df.groupby("customer_id").agg({
        "edad": "mean",
        "gender_code": "first",
        "total_compras": "first",
        "gasto_promedio": "first",
        "antiguedad_dias": "first",
        "categoria_frecuente_code": "first",
        "device_frecuente_code": "first",
        "preferred_price_bin_code": "first",
        "max_item_price": "first",
        "min_item_price": "first",
        "dias_promedio_compra": "first"
    }).reset_index()

    save_df(customer_feat, "customer_features.csv")
    return customer_feat

def process_df(df, training=True):
    """
    Investiga las siguientes funciones de SKlearn y determina si te son útiles
    - OneHotEncoder
    - StandardScaler
    - CountVectorizer
    - ColumnTransformer
    """
    numeric_features = [
        "edad", "total_compras", "gasto_promedio",
        "antiguedad_dias", "max_item_price", "min_item_price", "dias_promedio_compra"
    ]
    categorical_features = [
        "gender_code", "categoria_frecuente_code",
        "device_frecuente_code", "preferred_price_bin_code"
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )

    savepath = Path(DATA_DIR) / "preprocessor.pkl"
    if training:
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)

    processed_df = pd.DataFrame(
        processed_array,
        columns=numeric_features + list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features))
    )
    return processed_df

def preprocess(raw_df, training=False):
    processed_df = process_df(raw_df, training)
    return processed_df

def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    X = df_to_numeric(customer_feat)
    y = train_df["label"] if "label" in train_df.columns else None
    return X, y

def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features.csv")
    X_test = df_to_numeric(customer_feat)
    return X_test

if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
