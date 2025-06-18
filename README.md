# Proyek Akhir: Sistem Prediksi Dropout Mahasiswa

## Business Understanding

**Jaya Jaya Institut** menghadapi tantangan besar dalam mempertahankan tingkat kelulusan mahasiswa. Dropout mahasiswa dapat berdampak negatif pada kinerja institusi, kualitas lulusan, serta stabilitas keuangan kampus. Oleh karena itu, dibutuhkan sistem prediksi dropout yang mampu membantu tim akademik untuk:

* Mengidentifikasi mahasiswa yang berpotensi dropout secara dini.
* Memberikan rekomendasi tindakan intervensi yang tepat sasaran.
* Mengoptimalkan sumber daya akademik dan finansial.

### Permasalahan Bisnis

* Tingginya tingkat dropout mahasiswa yang berdampak pada citra institusi.
* Tidak adanya sistem monitoring yang proaktif untuk mendeteksi mahasiswa berisiko dropout.
* Sulitnya pengambilan keputusan berbasis data dalam proses intervensi akademik.

### Cakupan Proyek

* Melakukan eksplorasi data (EDA) secara komprehensif.
* Membangun model prediksi klasifikasi status mahasiswa menggunakan Random Forest dan XGBoost.
* Mengembangkan dashboard interaktif berbasis Streamlit.
* Menyediakan sistem rekomendasi action items berbasis hasil prediksi.

### Persiapan

**Sumber data:** `data.csv` (berisi data akademik, demografi, keuangan, dan histori perkuliahan mahasiswa).

**Setup environment:**

```
# Clone repository (opsional)
git clone <repository-url>
cd project-folder

# Membuat virtual environment
python -m venv .venv

# Aktivasi (Windows)
.\.venv\Scripts\activate

# Aktivasi (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Menjalankan aplikasi Streamlit
streamlit run app.py
```

**Catatan:** Pastikan seluruh file (`app.py`, `data.csv`, `dropout_rf_model.pkl`, `dropout_xgb_model.pkl`, `ordinal_encoder.pkl`, `label_encoder.pkl`) berada dalam root direktori yang sama.

---

## Business Dashboard

Dashboard interaktif dibangun menggunakan **Streamlit** yang terdiri dari:

1️⃣ **📊 Data Visualization**

* Visualisasi korelasi antar fitur.
* Deteksi outlier menggunakan IQR.
* Distribusi fitur penting (Admission Grade, Age at Enrollment, Gender, Course).

2️⃣ **🎯 Prediction**

* Form input interaktif untuk memasukkan data mahasiswa.
* Pilihan model: Random Forest atau XGBoost.
* Output prediksi status (Graduate, Enrolled, Dropout), confidence score, dan rekomendasi tindakan.

3️⃣ **📋 Recommendations**

* Hasil prediksi keseluruhan dataset.
* Filter multi-kriteria: Gender, Scholarship, Debtor, Actual Status, Predicted Status.
* Dynamic Recommended Actions berbasis prediksi dan status aktual.

---

## Menjalankan Sistem Machine Learning

Model machine learning yang digunakan telah dilatih dengan algoritma:

* Random Forest Classifier
* XGBoost Classifier
* Hyperparameter tuning dilakukan pada tahap training.
* Encoding kategori menggunakan OrdinalEncoder dan LabelEncoder.

Untuk menjalankan prototipe sistem prediksi:

```
streamlit run app.py
```

---

## Conclusion

Dengan diterapkannya sistem prediksi dropout berbasis machine learning, Jaya Jaya Institut kini memiliki:

* Sistem monitoring dini mahasiswa berisiko tinggi.
* Rekomendasi intervensi berbasis data.
* Alat bantu pengambilan keputusan manajemen akademik.
* Potensi peningkatan retensi dan kesuksesan studi mahasiswa.

### Rekomendasi Action Items

* Melakukan monitoring berkala terhadap mahasiswa dengan prediksi dropout saat masih berstatus enrolled.
* Memberikan mentoring akademik secara personal bagi mahasiswa berisiko.
* Meninjau kembali skema bantuan keuangan (financial aid) bagi mahasiswa yang menjadi debtor.
* Menyediakan layanan konseling psikologis untuk mahasiswa dengan tingkat stress akademik tinggi.
* Mengembangkan program re-entry dan alumni tracking untuk mahasiswa yang sudah dropout.
* Melakukan retraining model secara berkala dengan data terbaru agar akurasi prediksi tetap optimal.
