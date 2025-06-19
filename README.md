# Proyek Akhir: Sistem Prediksi Dropout siswa

## Business Understanding

Jaya Jaya Institut menghadapi tantangan besar dalam mempertahankan tingkat kelulusan siswa. Dropout siswa dapat berdampak negatif pada kinerja institusi, kualitas lulusan, serta stabilitas keuangan kampus. Oleh karena itu, dibutuhkan sistem prediksi dropout yang mampu membantu tim akademik untuk:

* Mengidentifikasi siswa yang berpotensi dropout secara dini.
* Memberikan rekomendasi tindakan intervensi yang tepat sasaran.
* Mengoptimalkan sumber daya akademik dan finansial.

### Permasalahan Bisnis

* Tingginya tingkat dropout siswa yang berdampak pada citra institusi.
* Tidak adanya sistem monitoring yang proaktif untuk mendeteksi siswa berisiko dropout.
* Sulitnya pengambilan keputusan berbasis data dalam proses intervensi akademik.

### Cakupan Proyek

* Melakukan eksplorasi data (EDA) secara komprehensif.
* Membangun model prediksi klasifikasi status siswa menggunakan Random Forest dan XGBoost.
* Mengembangkan dashboard interaktif berbasis Streamlit.
* Menyediakan sistem rekomendasi action items berbasis hasil prediksi.

### Persiapan

**Sumber data:** [Link Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Setup environment:**

```bash
# Clone repository
git clone https://github.com/Haiqelhakeem/student-performance-prediction.git

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

Dashboard interaktif dibangun menggunakan **Streamlit** yang berfungsi sebagai *Decision Support System* berbasis Machine Learning untuk membantu manajemen kampus memonitor potensi dropout mahasiswa secara proaktif. Dashboard terdiri dari 3 fitur utama berikut:

---

### 1️⃣ 📊 Data Visualization (Eksplorasi Data)

- Menampilkan **preview dataset** lengkap dengan `dataframe head()`.
- Visualisasi **distribusi kelas target** (Graduate, Enrolled, Dropout).
- **Correlation Heatmap** yang berfokus pada fitur-fitur yang paling berpengaruh terhadap dropout.
- Deteksi **outlier** menggunakan metode Interquartile Range (IQR) pada fitur-fitur yang paling berkorelasi.
- Visualisasi distribusi dari fitur-fitur kunci, antara lain:
  - `Admission Grade` (Nilai masuk mahasiswa)
  - `Age at Enrollment` (Usia saat masuk kuliah)
  - `Gender` (Jenis kelamin)
  - `Course` (Jurusan)
- Visualisasi **sebaran dropout berdasarkan gender dan jurusan** untuk membantu identifikasi kelompok risiko.

---

### 2️⃣ 🎯 Prediction (Prediksi Dropout Siswa)

- Form input interaktif bagi user untuk melakukan simulasi prediksi status mahasiswa secara individual.
- User dapat memilih model yang digunakan:  
  - `Random Forest Classifier` atau `XGBoost Classifier`.
- Sistem memproses data input melalui preprocessing dan encoding otomatis.
- Output yang dihasilkan:
  - Prediksi status (`Graduate`, `Enrolled`, `Dropout`).
  - Skor confidence prediksi dalam persentase.
  - Rekomendasi tindakan intervensi berbasis hasil prediksi.
- Cocok digunakan oleh tim akademik untuk mengevaluasi kondisi individu mahasiswa secara real-time.

---

### 3️⃣ 📋 Recommendations (Actionable Intervention Recommendations)

- Sistem melakukan inferensi otomatis pada seluruh dataset secara batch.
- Terdapat fitur **multi-filter interaktif** untuk eksplorasi hasil prediksi:
  - `Gender`, `Scholarship Holder`, `Debtor`, `Actual Status`, dan `Predicted Status`.
- Setiap baris data mahasiswa yang diprediksi diberikan **Recommended Action** secara otomatis berdasarkan kombinasi status aktual dan hasil prediksi:
  - 🧑‍🏫 `Academic Mentoring`
  - 🧠 `Counseling Support`
  - 💰 `Financial Aid Review`
  - 📞 `Alumni Follow-up`
  - ✅ `Continuous Monitoring`
- Fitur ini memungkinkan tim manajemen kampus melakukan pengambilan keputusan berbasis data secara cepat dan presisi.

**🔗 Link Akses Dashboard Streamlit:**

👉 [Student Performance Prediction Dashboard](https://student-performance-pred-haiqelhakeem.streamlit.app/)

---

## Menjalankan Sistem Machine Learning

Model machine learning yang digunakan telah dilatih dengan algoritma:

* Random Forest Classifier
* XGBoost Classifier
* Hyperparameter tuning dilakukan pada tahap training.
* Encoding kategori menggunakan OrdinalEncoder dan LabelEncoder.

Untuk menjalankan prototipe sistem prediksi:

```bash
streamlit run app.py
```

---

## Conclusion

Dengan diterapkannya sistem **Prediksi Dropout Mahasiswa berbasis Machine Learning**, Jaya Jaya Institut kini memiliki infrastruktur *Data-Driven Decision Support System* yang memberikan manfaat nyata bagi manajemen kampus, antara lain:

- 🧠 Faktor Utama Dropout:
  - Prestasi akademik semester 1 & 2 (grade & approved units)
  - Status keuangan: Debtor & Tuition Fees Up To Date
  - Tidak menerima beasiswa
  - Demografi: Laki-laki, Usia Muda

- 🧑‍🏫 Karakteristik Mahasiswa Dropout:
  - Nilai akademik rendah sejak awal
  - Jumlah mata kuliah lulus sedikit
  - Memiliki tunggakan pembayaran
  - Bukan penerima beasiswa
  - Usia muda dan cenderung laki-laki

- ✅ Solusi Paling Efektif:

  - 🧠 **Academic Early Intervention Program**  
    Deteksi dini mahasiswa dengan nilai rendah semester 1 & 2
    
  - 💰 **Financial Aid Prioritization**
    Beasiswa atau penjadwalan ulang pembayaran untuk mahasiswa berprestasi namun memiliki kendala finansial
    
  - 🧠 **Counseling & Academic Mentorship**
    Mentoring khusus untuk mahasiswa di bawah standar akademik
    
  - 💰 **Integrated Financial Monitoring System**  
    Integrasi data pembayaran dengan sistem monitoring akademik
  
  - 📊 **Sistem Monitoring Proaktif**  
    Kemampuan untuk mendeteksi mahasiswa yang menunjukkan potensi dropout sejak dini, sebelum masalah berkembang lebih jauh.
  
  - 🎯 **Sistem Rekomendasi Intervensi Otomatis**  
    Memberikan saran tindakan intervensi yang terpersonalisasi berdasarkan kombinasi hasil prediksi dan status aktual mahasiswa.
  
  - 📋 **Pendukung Pengambilan Keputusan Manajemen Akademik**  
    Memberikan gambaran menyeluruh berbasis data untuk rapat manajemen, penentuan kebijakan akademik, serta pengalokasian sumber daya.
  
  - 📈 **Potensi Peningkatan Retensi Mahasiswa**  
    Dengan intervensi dini, diharapkan tingkat kelulusan meningkat, menekan jumlah dropout, serta menjaga kualitas lulusan dan reputasi institusi.
  
  - 🔄 **Sistem Berkelanjutan yang Dapat Dilatih Ulang**  
    Model dapat diretrain secara berkala seiring penambahan data baru sehingga menjaga akurasi prediksi tetap relevan dengan dinamika mahasiswa.

---

### Rekomendasi Action Items

Berikut beberapa rekomendasi langkah taktis yang dapat dilakukan oleh institusi berdasarkan hasil sistem ini:

#### 1. Intervensi Dini Terhadap Mahasiswa Enrolled dengan Prediksi Dropout

- Melakukan review akademik secara rutin.
- Melibatkan wali akademik untuk monitoring progres semester.
- Menyusun program pembinaan belajar atau kelas remedial.

#### 2. Peningkatan Dukungan Finansial (Debtor & Tuition Fees)

- Menyusun skema bantuan keuangan yang lebih fleksibel.
- Memberikan opsi penjadwalan ulang pembayaran biaya kuliah.

#### 3. Penguatan Layanan Konseling dan Psikologis

- Menyediakan sesi konseling akademik maupun non-akademik.
- Mengedukasi mahasiswa tentang manajemen stres dan keseimbangan belajar.

#### 4. Pengembangan Program Re-entry dan Alumni

- Membuka peluang re-registrasi bagi mahasiswa dropout.
- Melakukan tracking alumni untuk evaluasi efektivitas program pembinaan.

#### 5. Penguatan Sistem Monitoring Berkelanjutan

- Memastikan tim manajemen secara aktif menggunakan dashboard prediksi sebagai tools operasional.
- Memanfaatkan fitur multi-filter pada dashboard untuk analisis populasi risiko spesifik (misal: per jurusan, gender, beasiswa, dll).

#### 6. Pengembangan Model Secara Berkelanjutan

- Mengupdate dataset setiap semester.
- Melakukan retraining model minimal 1 kali per tahun agar akurasi sistem terjaga mengikuti dinamika data terbaru.

