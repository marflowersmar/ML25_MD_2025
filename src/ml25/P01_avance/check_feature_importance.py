import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_processing import read_train_data

print("🔍 Cargando datos procesados...")
X, y = read_train_data()
print(f"✅ Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características.")

# Entrenar un RandomForest básico para diagnosticar
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X, y)

# Importancias de características
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(25)
print("\n📊 TOP 25 características por importancia:")
print(top_features)

# Correlación con el label
correlations = X.assign(label=y).corr(numeric_only=True)["label"].sort_values(ascending=False)
print("\n⚠️ TOP 15 variables más correlacionadas con el label:")
print(correlations.head(15))

print("\n✅ Diagnóstico completado.")
