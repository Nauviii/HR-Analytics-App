# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

**Latar Belakang**  
Jaya Jaya Maju adalah perusahaan multinasional di bidang Edutech yang berdiri sejak tahun 2000 dengan lebih dari 1.000 karyawan tersebar di seluruh Indonesia.  
Meskipun perusahaan sudah berkembang pesat, tingkat **attrition rate** (rasio karyawan keluar terhadap total karyawan) perusahaan cukup tinggi, mencapai **>10%**.  
Tingginya attrition rate berpotensi meningkatkan biaya perekrutan, pelatihan, dan mengganggu produktivitas perusahaan.  
Untuk itu, manajemen HR ingin memahami faktor-faktor yang memengaruhi keputusan karyawan untuk keluar dan mengembangkan strategi retensi yang tepat.  

---

## Permasalahan Bisnis
- Mengidentifikasi faktor-faktor utama penyebab tingginya attrition rate.  
- Mengatasi **data imbalance** dalam dataset attrition.  
- Membuat model prediksi attrition karyawan berbasis Machine Learning.  
- Menyediakan insight berbasis data untuk membantu pengambilan keputusan manajemen.  

---

## Cakupan Proyek
1. **Data Preprocessing & Feature Engineering**  
   - Membersihkan data, menangani missing values, encoding fitur kategorikal, dan scaling.  
2. **Exploratory Data Analysis (EDA)**  
   - Visualisasi dan analisis statistik untuk menemukan pola dan insight.  
3. **Model Machine Learning**  
   - Membangun model klasifikasi untuk memprediksi attrition karyawan.  
4. **Evaluasi Model**  
   - Menggunakan metrik evaluasi seperti Confusion Matrix, Classification Report, dan ROC-AUC.  
5. **Dashboard Interaktif**  
   - Membuat aplikasi analisis berbasis **Streamlit** untuk memudahkan eksplorasi data dan insight.  

---

## Persiapan
### 🔗 Dataset
- **Sumber Data**: [Employee Data - Dicoding](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv)  
- **Jumlah Data**: 1.470 karyawan dengan tingkat attrition **12.18%** (179 karyawan keluar).  

---

### 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Streamlit** (Pembuatan Dashboard)
- **Joblib** (Model Serialization)
- **GitHub** (Version Control)

---

### ⚙️ Persiapan & Setup
Clone repository dan install dependencies:
```bash
git clone https://github.com/Nauviii/Data-Science-HR-Analytics.git
cd Data-Science-HR-Analytics
pip install -r requirements.txt
streamlit run app.py
```

## Business Dashboard

🔗 Akses Dashboard: https://hr-analytics-app.streamlit.app/

Dashboard ini adalah aplikasi interaktif berbasis Streamlit yang membantu stakeholder menganalisis tingkat attrition. Dashboard mencakup:
1. EDA (Exploratory Data Analysis) – Visualisasi distribusi data dan korelasi antar fitur.
2. Model Results – Hasil prediksi attrition dengan metrik evaluasi.
3. Insights & Recommendations – Temuan utama dan rekomendasi actionable.

## Conclusion
📋 Ringkasan Proyek

Proyek ini mengembangkan end-to-end pipeline data science untuk memprediksi attrition karyawan dan menyajikan insight melalui dashboard interaktif.
- Data: 1.470 karyawan, attrition rate 12.18%.
- Hasil: Model Machine Learning teroptimasi dengan PCA untuk dimensionality reduction, tersimpan dalam format joblib.
- Output: Dashboard analitik HR berbasis Streamlit.

## Rekomendasi Action Items (Optional)
1. Peningkatan Kepuasan Kerja & Lingkungan
        - Rutin melakukan employee engagement survey.
        - Program kesejahteraan dan komunikasi terbuka.

2. Program Retensi Karyawan Muda
        - Mentorship, pelatihan soft skill, dan rotasi kerja.

3. Optimasi Manajemen R&D
        - Evaluasi beban kerja, pemberian reward untuk inovasi.
   
4. Dukungan Transisi Manajemen
        - Onboarding manager baru & sesi check-in rutin.

5. Perbaikan Kompensasi & Benefit
        - Benchmarking gaji dan penawaran benefit non-finansial.

