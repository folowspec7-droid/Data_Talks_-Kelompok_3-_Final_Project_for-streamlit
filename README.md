# 🌦️ Rain Predictor (Streamlit + XGBoost)

Prediksi peluang hujan besok dari data cuaca hari ini (dataset weatherAUS).

## Menjalankan lokal
    pip install -r requirements.txt
    python train_model.py        # opsional, melatih XGBoost -> model.joblib
    streamlit run app.py

Jika `model.joblib` belum ada (atau tidak cocok dengan versi library di server),
aplikasi otomatis melatih XGBoost dari `weatherAUS.csv` saat pertama dibuka
(sekitar 1 menit, lalu di-cache selama server hidup).

## Deploy ke Streamlit Community Cloud
1. Upload isi folder ini ke GitHub. Sertakan `weatherAUS.csv`; `model.joblib` boleh
   ikut jika dilatih dengan versi xgboost/scikit-learn yang sama dengan requirements.
2. Di share.streamlit.io pilih repo, main file: `app.py`.
3. Agar versi sama persis antara latih dan deploy, pin di requirements.txt, mis.
   `xgboost==<versi lokal Anda>` dan `scikit-learn==<versi lokal Anda>`.
