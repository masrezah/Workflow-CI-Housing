import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# 1. Load Data
# Kita gunakan path relatif agar aman saat dijalankan robot GitHub
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "clean_housing_data.csv")

print(f"Memuat data dari: {data_path}")
df = pd.read_csv(data_path)

X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Training Model (Parameter Fixed untuk CI)
# Robot akan menggunakan parameter ini
n_estimators = 100
max_depth = 10

print(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X_train, y_train)

# 3. Evaluasi
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Hasil -> MAE: {mae}, MSE: {mse}, R2: {r2}")

# 4. Simpan Hasil ke File (Syarat Skilled: Menyimpan Artefak)
# Kita simpan metrik ke file teks agar bisa di-push balik ke GitHub
metrics_file = os.path.join(current_dir, "metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"R2 Score: {r2}\n")

print(f"Metrik berhasil disimpan ke {metrics_file}")

# 5. MLflow Tracking (Opsional untuk CI, tapi baik untuk standar)
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)
mlflow.log_metric("mae", mae)
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.sklearn.log_model(rf, "model")