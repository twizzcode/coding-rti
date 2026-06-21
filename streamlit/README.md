# Software Defect Prediction Dashboard

Aplikasi Streamlit untuk menjalankan eksperimen **Software Defect Prediction** berdasarkan paper:

**Comparative Analysis of Feature Selection Methods and Machine Learning Algorithms for Software Defect Prediction Using NASA MDP Datasets**

## Fitur

- Upload dataset CSV
- Auto-detect kolom target defect
- Dataset overview
- Preprocessing
- Feature Selection:
  - No Feature Selection
  - Chi-Square
  - Mutual Information
  - ANOVA F-test
- Machine Learning:
  - Random Forest
  - Support Vector Machine
  - XGBoost
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC
  - MCC
- Confusion Matrix
- Ranking fitur
- Run all scenarios
- Download hasil eksperimen CSV

## Cara Menjalankan

### 1. Masuk folder project

```bash
cd sdp_streamlit_app
```

### 2. Buat virtual environment

```bash
python -m venv .venv
```

Aktifkan environment:

Linux/Mac:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependency

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Buka URL yang muncul, biasanya:

```bash
http://localhost:8501
```

## Format Dataset

Dataset harus CSV dan memiliki:

- Kolom fitur numerik, misalnya software metrics.
- Satu kolom target/label, misalnya:
  - `defects`
  - `defect`
  - `bug`
  - `label`
  - `class`

Contoh target:

```csv
loc,v(g),ev(g),iv(g),n,v,l,d,i,e,b,t,defects
10,2,1,2,50,200,0.5,10,20,2000,0.06,111,0
80,10,8,9,300,1200,0.1,50,24,60000,0.40,3333,1
```

## Catatan

Sample dataset di aplikasi hanya data dummy untuk tes tampilan. Untuk eksperimen sebenarnya, gunakan dataset NASA MDP asli seperti JM1, KC1, PC1, CM1, dan KC3.
