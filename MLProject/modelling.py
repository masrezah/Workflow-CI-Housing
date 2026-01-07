import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Dataset Bersih
# Pastikan path ini sesuai dengan struktur folder Anda
df = pd.read_csv('housing_preprocessing/clean_housing_data.csv')

# Pisahkan Fitur (X) dan Target (y)
# Sesuaikan 'MEDV' jika nama kolom target di dataset Anda berbeda
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =================================================================
# BAGIAN PENTING: AKTIFKAN AUTOLOG
# =================================================================
# Ini syarat mutlak untuk Kriteria Basic.
# Autolog akan otomatis mencatat parameter, metrik, dan menyimpan model.
mlflow.autolog()

# =================================================================
# TRAINING MODEL DENGAN MLFLOW
# =================================================================
with mlflow.start_run():
    # Definisikan Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Latih Model
    # Saat .fit() dipanggil, MLflow autolog akan bekerja otomatis
    rf.fit(X_train, y_train)
    
    # Evaluasi (Opsional, karena autolog biasanya sudah menghitung ini)
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained. MSE: {mse}")

    # TIDAK PERLU ada baris:
    # mlflow.log_param(...)  <- HAPUS
    # mlflow.log_metric(...) <- HAPUS
    # mlflow.sklearn.log_model(...) <- HAPUS (Autolog sudah menangani ini)

print("Training selesai. Cek MLflow UI untuk artefak.")
