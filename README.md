# Deteksi Cyberbullying Bahasa Indonesia: Analisis Komparatif Model Machine Learning, Deep Learning, dan Transformer

Repository ini berisi proyek penelitian komparatif untuk mendeteksi tindakan cyberbullying pada teks media sosial berbahasa Indonesia. Evaluasi dilakukan terhadap beberapa model klasifikasi teks, mencakup model Machine Learning konvensional, Deep Learning sekuensial, dan model berbasis Transformer.

---

## 1. Latar Belakang
Perkembangan teknologi informasi telah membawa dampak signifikan terhadap interaksi sosial digital. Di balik kemudahan yang ditawarkan, fenomena negatif seperti cyberbullying semakin marak terjadi. Cyberbullying merupakan tindakan perundungan berulang melalui platform digital (media sosial, pesan instan, forum online) yang dapat menyebabkan dampak psikologis serius bagi korban, mulai dari kecemasan dan depresi hingga tindakan fatal (Fadli & Hidayatullah, 2021).

Survei oleh Asosiasi Penyelenggara Jasa Internet Indonesia (APJII) terhadap 3.077 siswa (usia 13-18 tahun, jenjang SMP/SMA) di 34 provinsi menunjukkan bahwa 1.895 siswa (45,33%) mengaku pernah menjadi korban cyberbullying, sedangkan 1.182 siswa (38,41%) menjadi pelaku (Tahir & Sugianto, 2024).

Mengingat volume data yang besar dan penyebaran informasi yang cepat, deteksi manual menjadi tidak efektif. Diperlukan sistem otomatis yang efisien dan akurat untuk mengklasifikasi teks cyberbullying. Tantangan utamanya terletak pada kompleksitas bahasa gaul, sindiran, serta ekspresi emosional khas Indonesia.

Untuk mengatasi ini, penelitian ini membandingkan empat representasi model:
1.  **Transformer**: **IndoBERT** (`indolem/indobert-base-uncased`) dan **DistilBERT** (`cahya/distilbert-base-indonesian`). DistilBERT merupakan versi ringan (student model) dari BERT (teacher model) yang dilatih dengan metode *knowledge distillation* guna mempercepat inferensi dengan penurunan performa minimal (Basbeth, 2024).
2.  **Deep Learning**: **Bidirectional Gated Recurrent Unit (Bi-GRU)** yang dipadukan dengan FastText (`cc.id.300.vec`). Karakter sekuensial dua arah (bidirectional) memproses teks dari depan dan belakang guna menangkap ketergantungan antar kata tanpa parameter sebanyak LSTM (Cho et al., 2014).
3.  **Machine Learning Tradisional**: **Support Vector Machine (SVM)** dengan ekstraksi fitur TF-IDF sebagai baseline pembanding untuk memvalidasi apakah kompleksitas komputasi model deep learning sebanding dengan peningkatan performa yang dihasilkan (Yarsasi & Iskoko, 2025).

---

## 2. Rumusan Masalah & Ruang Lingkup

### Rumusan Masalah
1.  Bagaimana mengembangkan sistem deteksi cyberbullying pada media sosial yang mampu beroperasi secara efisien tanpa mengorbankan akurasi?
2.  Bagaimana penerapan fine-tuning meningkatkan akurasi deteksi teks bermuatan cyberbullying?
3.  Metode fine-tuning/pooling apa yang menghasilkan performa akurasi terbaik?
4.  Bagaimana hasil perbandingan metrik ukuran model, waktu training, akurasi, dan F1-score pada model-model tersebut?
5.  Model manakah yang menawarkan efisiensi prediksi teks cyberbullying terbaik?

### Ruang Lingkup
*   **Bahasa**: Dataset hanya menggunakan teks berbahasa Indonesia.
*   **Sumber Data**: Dataset sekunder terbuka yang digabungkan dari 3 sumber:
    1.  *Cyberbullying Bahasa Indonesia* (Kaggle, oleh Cita Tiara Hanni, 2021).
    2.  *Cyberbullying Dataset* (Hugging Face, oleh aditdwi123, 2024).
    3.  *Komentar Instagram Cyberbullying* (GitHub, oleh rizalespe, 2019).
*   **Model Evaluasi**: IndoBERT, DistilBERT, Bi-GRU, dan SVM.

---

## 3. Metode Penelitian & Alur Pra-pemrosesan

Penelitian ini berjalan melalui beberapa tahapan sistematis:
1.  **Identifikasi Masalah & Studi Literatur**: Menelaah kelebihan/kekurangan model BERT, DistilBERT, Bi-GRU, dan SVM dalam klasifikasi teks.
2.  **Pra-pemrosesan Data**:
    *   **Pembersihan Data (Text Cleaning)**: Menghapus URL, tanda baca, angka, simbol, tag HTML, dan mention username (`@USERNAME`).
    *   **Normalisasi Teks**: Mengubah teks menjadi huruf kecil (*lowercasing*), menyelaraskan slang/singkatan menggunakan kamus normalisasi ([kamus_singkatan.csv](file:///home/ai-agent/workspace/Indonesian-Cyberbullying-Detection-with-Distilbert/Dataset/kamus_singkatan.csv)), serta memfilter stopword ([stopwordbahasa.csv](file:///home/ai-agent/workspace/Indonesian-Cyberbullying-Detection-with-Distilbert/Dataset/stopwordbahasa.csv)).
    *   **Tokenisasi**: Memecah kalimat menjadi token menggunakan NLTK tokenizer atau spaCy tokenizer untuk Bahasa Indonesia.
    *   **Binary Label Encoding**: Mengubah kelas target menjadi nilai numerik (`0` untuk non-cyberbullying dan `1` untuk cyberbullying).
3.  **Optimisasi Hyperparameter**: Menerapkan pencarian otomatis menggunakan framework **Optuna** dengan algoritma *Tree-structured Parzen Estimator* (TPE).
4.  **Prediksi dan Evaluasi**: Melatih dan menguji model dengan metrik Akurasi, Presisi, Recall, F1-Score, serta waktu inferensi rata-rata.

---

## 4. Hasil Evaluasi Model

Berikut adalah tabel hasil komparasi performa dan efisiensi prediksi model pada dataset uji:

| Arsitektur Model | Representasi Fitur / Embedding | Skema Validasi | Akurasi Uji | Waktu Inferensi Rata-rata / Batch (s) |
| :--- | :--- | :--- | :---: | :---: |
| **IndoBERT (Fine-tuned)** | AutoTokenizer + Transformer Embedding | Train/Val/Test Split | **90.82%** | ~0.150 s |
| **Bi-GRU (Max Pooling)** | DistilBERT Embedding | 5-Fold Cross-Validation | **89.37%** | ~0.112 s |
| **SVM (Linear)** | TF-IDF Vectorizer | 5-Fold Cross-Validation | **88.16%** | **0.004 s** |
| **Bi-GRU** | spaCy Tokenizer + FastText | No K-Fold Split | **87.92%** | ~0.081 s |
| **Bi-GRU** | spaCy Tokenizer + FastText | 5-Fold Cross-Validation | **86.96%** | ~0.079 s |
| **SVM** | spaCy Tokenizer + FastText | 5-Fold Cross-Validation | **86.96%** | ~0.006 s |
| **Bi-GRU** | NLTK Tokenizer + FastText | 5-Fold Cross-Validation | **86.23%** | ~0.053 s |
| **SVM** | NLTK Tokenizer + FastText | 5-Fold Cross-Validation | **86.23%** | ~0.006 s |
| **Bi-GRU (Mean Pooling)** | DistilBERT Embedding | 5-Fold Cross-Validation | **85.51%** | ~0.112 s |

### Analisis Hasil
*   **Performa Tertinggi**: Model **IndoBERT** yang di-*fine-tune* menghasilkan akurasi tertinggi sebesar **90.82%** (F1-score macro 0.9021). Hal ini dikarenakan kemampuannya dalam memahami representasi semantik kalimat secara dua arah (*bidirectional*).
*   **Analisis Efisiensi**: Meskipun IndoBERT memberikan performa terbaik, **SVM berbasis TF-IDF** terbukti menjadi alternatif baseline yang sangat kompetitif. Model SVM mencapai akurasi **88.16%** dengan waktu inferensi batch sebesar **0.004 detik** (sekitar 37x lebih cepat dibandingkan IndoBERT).
*   **Analisis Pra-pemrosesan**: Penggunaan tokenisasi berbasis **spaCy** secara konsisten menghasilkan performa klasifikasi yang sedikit lebih baik dibandingkan **NLTK** ketika digabungkan dengan representasi embedding FastText pada model sekuensial Bi-GRU.

---

## 5. Struktur Repositori

```directory
.
├── Dataset/                     # Repositori data sekunder
│   ├── kamus_singkatan.csv      # Kamus normalisasi kata gaul/singkatan
│   ├── stopwordbahasa.csv       # Daftar stopword Bahasa Indonesia
│   └── Pre-Processed Dataset/   # Dataset yang telah dibersihkan
├── notebooks/                   # Notebook utama hasil implementasi
│   ├── BERT_DistilBERT_Training_Evaluation.ipynb  # Notebook fine-tuning Transformer (IndoBERT/DistilBERT)
│   ├── bi_gru/                  # Notebook model klasifikasi Bi-GRU
│   │   ├── Bi_GRU_nltk_FastText_KFold.ipynb
│   │   ├── MaxPooling_Bi_GRU_DistilBERT_KFold.ipynb
│   │   └── ...
│   └── svm/                     # Notebook model SVM
│       ├── SVM_TFIDF_KFold.ipynb
│       └── ...
├── archive/                     # Folder arsip pengujian parameter & model non-final
│   ├── non_final_bigru_added_dataset/
│   ├── inference_time_testing_svm/
│   └── ...
└── README.md
```

---

## 6. Panduan Penggunaan

### Instalasi Lokal
1.  **Clone repositori**:
    ```bash
    git clone https://github.com/WilliamAxelC/Indonesian-Cyberbullying-Detection-with-Distilbert.git
    cd Indonesian-Cyberbullying-Detection-with-Distilbert
    ```
2.  **Instalasi dependensi**:
    Instal pustaka Python utama yang diperlukan (seperti PyTorch, Transformers, Optuna, Scikit-Learn, Pandas, spaCy, NLTK, dan ipywidgets):
    ```bash
    pip install torch transformers[torch] accelerate optuna scikit-learn pandas spacy nltk ipywidgets
    ```
3.  **Unduh file FastText (Khusus Notebook FastText)**:
    Unduh representasi vektor FastText bahasa Indonesia (`cc.id.300.vec.gz`) dari laman [FastText Crawl Vectors](https://fasttext.cc/docs/en/crawl-vectors.html), ekstrak file tersebut, dan letakkan file `cc.id.300.vec` di dalam root repositori ini.

### Menjalankan di Google Colab
Seluruh notebook di dalam folder `notebooks/` telah diintegrasikan dengan kode deteksi lingkungan. Ketika dijalankan di Google Colab, notebook akan mendeteksi lingkungan tersebut dan meminta *mounting* Google Drive agar dapat memproses data secara dinamis.
