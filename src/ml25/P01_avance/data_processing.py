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
DATA_DIR = CURRENT_FILE.parent

def read_csv(filename: str):
    data_root = DATA_DIR.parent / "datasets" / "customer_purchases"
    file = data_root / f"{filename}.csv"
    return pd.read_csv(file)

def save_df(df, filename: str):
    save_path = os.path.join(DATA_DIR, "customer_features.csv")
    df.to_csv(save_path, index=False)
    print(f"DataFrame saved to {save_path}")

def _numeric_customer_id(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    extr = s.str.extract(r"(\d+)", expand=False)
    has = extr.notna()
    out = pd.to_numeric(extr.where(has, np.nan), errors="coerce")
    nodigit = s[~has]
    if not nodigit.empty:
        uniq = sorted(nodigit.unique())
        mp = {v: i + 1 for i, v in enumerate(uniq)}
        out.loc[~has] = nodigit.map(mp).astype(float)
    return out.fillna(-1).astype(int)

def extract_customer_features(train_df):
    df = train_df.copy()

    df["customer_date_of_birth"] = pd.to_datetime(df["customer_date_of_birth"], errors="coerce")
    df["customer_signup_date"] = pd.to_datetime(df["customer_signup_date"], errors="coerce")
    df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
    ref_ts = pd.Timestamp(DATA_COLLECTED_AT)

    df["edad"] = np.floor((ref_ts - df["customer_date_of_birth"]).dt.days / 365.25)
    df["antiguedad_dias"] = (ref_ts - df["customer_signup_date"]).dt.days
    df["customer_gender"] = df["customer_gender"].astype(str).str.lower()
    df["gender_code"] = df["customer_gender"].map({"female": 0, "male": 1}).fillna(-1).astype(int)
    df["customer_id_num"] = _numeric_customer_id(df["customer_id"])

    cat_frec = (
        df.groupby("customer_id_num")["item_category"]
          .agg(lambda s: s.mode(dropna=True).iat[0] if not s.mode(dropna=True).empty else np.nan)
          .reset_index(name="categoria_frecuente")
    )
    dev_frec = (
        df.groupby("customer_id_num")["purchase_device"]
          .agg(lambda s: s.mode(dropna=True).iat[0] if not s.mode(dropna=True).empty else np.nan)
          .reset_index(name="device_frecuente")
    )
    pref = cat_frec.merge(dev_frec, on="customer_id_num", how="outer")
    pref["categoria_frecuente_code"] = pref["categoria_frecuente"].astype("category").cat.codes.astype(int)
    pref["device_frecuente_code"] = pref["device_frecuente"].astype("category").cat.codes.astype(int)

    price_stats = (
        df.groupby("customer_id_num", as_index=False)["item_price"]
          .agg(max_item_price="max", min_item_price="min", gasto_promedio="mean")
    )
    total_comp = (
        df.groupby("customer_id_num", as_index=False)["item_id"]
          .size().rename(columns={"size": "total_compras"})
    )

    med_price = (
        df.groupby("customer_id_num", as_index=False)["item_price"]
          .median().rename(columns={"item_price": "median_price"})
    )
    bins = [0, 500, 1000, 1500, 2000, 5000, np.inf]
    labels = [0, 1, 2, 3, 4, 5]
    med_price["preferred_price_bin_code"] = pd.cut(med_price["median_price"], bins=bins, labels=labels).astype(int)

    df_sorted = df.sort_values(["customer_id_num", "purchase_timestamp"])
    diffs = df_sorted.groupby("customer_id_num")["purchase_timestamp"].diff().dt.days
    dias_prom = (
        diffs.groupby(df_sorted["customer_id_num"])
             .mean().fillna(0).reset_index(name="dias_promedio_compra")
    )

    edad_ant = (
        df.groupby("customer_id_num", as_index=False)
          .agg(edad=("edad", "mean"), antiguedad_dias=("antiguedad_dias", "mean"))
    )

    gender_mode = (
        df.groupby("customer_id_num")["gender_code"]
          .agg(lambda s: s.mode(dropna=True).iat[0] if not s.mode(dropna=True).empty else -1)
          .reset_index(name="gender_code")
    )

    colors = ["black","blue","green","orange","pink","red","white","yellow"]
    if "item_color" in df.columns:
        col_series = df["item_color"].astype(str).str.lower()
    else:
        title = df["item_title"].astype(str).str.lower()
        pattern = r"\b(" + "|".join(colors) + r")\b"
        col_series = title.str.extract(pattern, expand=False).fillna("unknown")
    color_counts = (
        pd.crosstab(df["customer_id_num"], col_series)
          .reindex(columns=colors, fill_value=0)
          .add_prefix("color_").add_suffix("_count")
          .reset_index()
    )

    cat_order = ["blouse","dress","jacket","jeans","shirt","shoes","skirt","slacks","suit","t-shirt"]
    cat_series = df["item_category"].astype(str).str.lower()
    cat_counts = (
        pd.crosstab(df["customer_id_num"], cat_series)
          .reindex(columns=cat_order, fill_value=0)
          .add_prefix("cat_").add_suffix("_count")
          .reset_index()
    )

    customer_feat = (
        price_stats
        .merge(total_comp, on="customer_id_num", how="left")
        .merge(pref[["customer_id_num","categoria_frecuente_code","device_frecuente_code"]], on="customer_id_num", how="left")
        .merge(med_price[["customer_id_num","preferred_price_bin_code"]], on="customer_id_num", how="left")
        .merge(dias_prom, on="customer_id_num", how="left")
        .merge(edad_ant, on="customer_id_num", how="left")
        .merge(gender_mode, on="customer_id_num", how="left")
        .merge(color_counts, on="customer_id_num", how="left")
        .merge(cat_counts, on="customer_id_num", how="left")
        .rename(columns={"customer_id_num": "customer_id"})
    )

    required_cols = [
        "customer_id","edad","gender_code","total_compras","gasto_promedio","antiguedad_dias",
        "categoria_frecuente_code",
        "color_black_count","color_blue_count","color_green_count","color_orange_count",
        "color_pink_count","color_red_count","color_white_count","color_yellow_count",
        "device_frecuente_code","preferred_price_bin_code","max_item_price","min_item_price",
        "dias_promedio_compra",
        "cat_blouse_count","cat_dress_count","cat_jacket_count","cat_jeans_count","cat_shirt_count",
        "cat_shoes_count","cat_skirt_count","cat_slacks_count","cat_suit_count","cat_t-shirt_count"
    ]
    for c in required_cols:
        if c not in customer_feat.columns:
            customer_feat[c] = 0
    customer_feat = customer_feat[required_cols]

    for c in customer_feat.columns:
        customer_feat[c] = pd.to_numeric(customer_feat[c], errors="coerce").fillna(0)

    save_df(customer_feat, "customer_features.csv")
    return customer_feat

def process_df(df, training=True):
    numeric_features = [
        "edad","total_compras","gasto_promedio","antiguedad_dias",
        "max_item_price","min_item_price","dias_promedio_compra"
    ]
    categorical_features = [
        "gender_code","categoria_frecuente_code","device_frecuente_code","preferred_price_bin_code"
    ]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features),
                      ("cat", categorical_transformer, categorical_features)],
        remainder="drop"
    )
    savepath = DATA_DIR / "preprocessor.pkl"
    if training:
        processed_array = preprocessor.fit_transform(df)
        joblib.dump(preprocessor, savepath)
    else:
        preprocessor = joblib.load(savepath)
        processed_array = preprocessor.transform(df)
    cat_names = list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_features))
    colnames = [f"num__{c}" for c in numeric_features] + [f"cat__{c}" for c in cat_names]
    processed_df = pd.DataFrame(processed_array, columns=colnames, index=df.index)
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
    """
    Lee el dataset generado por negative_generation.py
    """
    try:
        full_data_path = DATA_DIR / "train_df_full.csv"
        
        if not full_data_path.exists():
            full_data_path = Path("train_df_full.csv")
        
        if full_data_path.exists():
            full_data = pd.read_csv(full_data_path)
            
            y = full_data['label']
            X = full_data.drop(['customer_id_num', 'item_id_num', 'label'], axis=1, errors='ignore')
            
            print(f"Dataset de entrenamiento cargado: {X.shape}, Labels: {y.shape}")
            print(f"Distribucion de labels: {y.value_counts().to_dict()}")
            return X, y
        else:
            print("train_df_full.csv no encontrado, usando datos de ejemplo")
            X = pd.DataFrame(np.random.randn(1000, 10))
            y = pd.Series(np.random.randint(0, 2, 1000))
            return X, y
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        X = pd.DataFrame(np.random.randn(500, 8))
        y = pd.Series(np.random.randint(0, 2, 500))
        return X, y

def read_test_data():
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")
    X_test = df_to_numeric(customer_feat)
    return X_test

if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)
    print("\nGenerating customer_features.csv...")
    extract_customer_features(train_df)