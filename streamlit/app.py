# ============================================================
#  Software Defect Prediction - Streamlit App
#  Comparative Analysis of Feature Selection & ML Algorithms
#  NASA MDP Datasets (JM1, KC1, PC1, CM1, KC3)
# ============================================================

import io
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.io import arff as arff_io
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       mutual_info_classif)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2rem; font-weight:700; color:#1a3a5c; text-align:center; margin-bottom:.3rem; }
    .sub-title   { font-size:1rem; color:#555; text-align:center; margin-bottom:1.5rem; }
    .sec-header  { font-size:1.25rem; font-weight:600; color:#1f4e79;
                   border-left:4px solid #2e75b6; padding-left:.6rem; margin:1.2rem 0 .6rem; }
    .info-box    { background:#eef6ff; border-radius:8px; padding:1rem; border:1px solid #bcd8f5; }
    .best-box    { background:#e8f5e9; border-radius:8px; padding:1rem;
                   border:1px solid #a5d6a7; font-size:1.05rem; }
    .warn-box    { background:#fff8e1; border-radius:8px; padding:1rem; border:1px solid #ffe082; }
    div[data-testid="metric-container"] { background:#f7fbff; border-radius:8px;
                                          border:1px solid #d0e9ff; padding:.4rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("datasets", {}),
    ("results_df", None),
    ("summary_df", None),
    ("auto_loaded_files", []),
    ("autoload_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# CORE FUNCTIONS  (same logic as original .py)
# ─────────────────────────────────────────────

def read_arff_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Parse ARFF dari raw bytes dengan fallback encoding."""
    last_err = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            content = file_bytes.decode(enc)
            data, _ = arff_io.loadarff(io.StringIO(content))
            df = pd.DataFrame(data)
            return decode_bytes_dataframe(df)
        except Exception as e:
            last_err = e

    # fallback terakhir: tetap coba baca dengan karakter rusak diganti
    try:
        content = file_bytes.decode("utf-8", errors="replace")
        data, _ = arff_io.loadarff(io.StringIO(content))
        df = pd.DataFrame(data)
        return decode_bytes_dataframe(df)
    except Exception:
        raise last_err


def decode_bytes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah value bytes dari ARFF menjadi string biasa."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
            )
    return df


def read_uploaded_file(uploaded) -> pd.DataFrame | None:
    """Reader fleksibel untuk .arff, .csv, .xlsx, .xls."""
    name = uploaded.name.lower()
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()

    if name.endswith(".csv"):
        # Coba beberapa separator umum agar file CSV dari Excel juga aman
        for sep in [None, ",", ";", "\t"]:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            except Exception:
                continue
        raise ValueError("CSV tidak bisa dibaca. Cek delimiter atau encoding file.")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    if name.endswith(".arff"):
        return read_arff_bytes(raw)

    st.error(f"Format tidak didukung: {name}")
    return None




def get_data_dir() -> Path:
    """Folder data permanen. Default: folder data di sebelah app.py."""
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / "data"


def read_local_file(path: Path) -> pd.DataFrame | None:
    """Baca dataset dari file lokal di folder data/."""
    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".arff":
        return read_arff_bytes(raw)

    if suffix == ".csv":
        for sep in [None, ",", ";", "\t"]:
            try:
                return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            except Exception:
                continue
        raise ValueError("CSV tidak bisa dibaca. Cek delimiter atau encoding file.")

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(raw))

    return None


def load_datasets_from_data_folder(force: bool = False):
    """Auto-load semua dataset dari folder data/.

    Urutan kerja:
    1. Cek folder data/ di lokasi yang sama dengan app.py.
    2. Baca semua file .arff, .csv, .xlsx, .xls.
    3. Nama dataset dideteksi otomatis dari nama file.
    4. Dataset masuk ke st.session_state.datasets.
    """
    if st.session_state.get("autoload_done") and not force:
        return st.session_state.get("auto_loaded_files", [])

    data_dir = get_data_dir()
    loaded_files = []

    if not data_dir.exists():
        st.session_state.auto_loaded_files = []
        st.session_state.autoload_done = True
        return []

    supported_ext = {".arff", ".csv", ".xlsx", ".xls"}
    files = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_ext])

    for path in files:
        try:
            df = read_local_file(path)
            if df is not None and not df.empty:
                ds_name = infer_dataset_name(path.name, st.session_state.datasets.keys())
                st.session_state.datasets[ds_name] = df
                loaded_files.append({
                    "Dataset": ds_name,
                    "File": path.name,
                    "Baris": int(df.shape[0]),
                    "Kolom": int(df.shape[1]),
                    "Status": "Auto-loaded dari folder data",
                })
        except Exception as e:
            loaded_files.append({
                "Dataset": "-",
                "File": path.name,
                "Baris": 0,
                "Kolom": 0,
                "Status": f"Gagal dibaca: {e}",
            })

    st.session_state.auto_loaded_files = loaded_files
    st.session_state.autoload_done = True
    return loaded_files

def infer_dataset_name(filename: str, existing_names=None) -> str:
    """Ambil nama dataset otomatis dari nama file: JM1, KC1, PC1, CM1, KC3, dst."""
    existing_names = set(existing_names or [])
    stem = filename.rsplit(".", 1)[0].strip()
    normalized = stem.upper().replace(" ", "_")

    # Pola NASA MDP yang umum, termasuk nama seperti JM1(1).arff
    for key in ["JM1", "KC1", "PC1", "CM1", "KC3"]:
        if key in normalized:
            return key

    # Kalau bukan dataset NASA yang dikenali, pakai nama file tanpa ekstensi
    name = stem if stem else "Dataset"
    base = name
    i = 2
    while name in existing_names:
        name = f"{base}_{i}"
        i += 1
    return name


def detect_target_column(df: pd.DataFrame) -> str:
    """Deteksi kolom target secara fleksibel."""
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    priority = [
        "defective", "defects", "defect", "bug", "bugs", "fault", "faults",
        "label", "class", "target", "y", "is_defective", "is_buggy",
    ]
    for key in priority:
        if key in lower_map:
            return lower_map[key]

    # Jika ada nama kolom yang mengandung defect/bug/fault, gunakan itu
    for c in cols:
        cl = str(c).strip().lower()
        if any(token in cl for token in ["defect", "bug", "fault", "label", "class", "target"]):
            return c

    # Fallback: kolom terakhir
    return cols[-1]


def normalize_target(y: pd.Series) -> pd.Series:
    """Normalisasi target menjadi 0 = non-defect, 1 = defect."""
    y = y.apply(lambda x: x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x)

    # Target string/kategori/bool seperti Y/N, true/false, defective/non-defective
    if y.dtype == object or str(y.dtype) == "category" or y.dtype == bool:
        ys = y.astype(str).str.strip().str.lower()
        ys = ys.replace({"nan": np.nan, "none": np.nan, "?": np.nan, "": np.nan})

        mapping = {
            "y": 1, "yes": 1, "true": 1, "t": 1, "1": 1, "defect": 1,
            "defects": 1, "defective": 1, "bug": 1, "buggy": 1, "fault": 1,
            "faulty": 1, "error": 1, "erroneous": 1,
            "n": 0, "no": 0, "false": 0, "f": 0, "0": 0, "non-defect": 0,
            "non_defect": 0, "nondefect": 0, "not defective": 0, "clean": 0,
            "not_buggy": 0, "not buggy": 0, "not faulty": 0,
        }

        mapped = ys.map(mapping)
        if mapped.notna().sum() > 0 and mapped.isna().sum() == ys.isna().sum():
            return mapped

        # Fallback binary string: urutkan label, lalu label kedua dianggap 1
        non_null = ys.dropna()
        uniq = sorted(non_null.unique().tolist())
        if len(uniq) == 2:
            enc_map = {uniq[0]: 0, uniq[1]: 1}
            return ys.map(enc_map)

        # Fallback multi-class string: LabelEncoder, lalu >0 dianggap defect
        le = LabelEncoder()
        encoded = pd.Series(np.nan, index=ys.index, dtype="float")
        mask = ys.notna()
        encoded.loc[mask] = le.fit_transform(ys.loc[mask]).astype(float)
        return (encoded > 0).astype(float).where(mask, np.nan)

    # Target numerik: nilai > 0 dianggap defect, 0 dianggap non-defect
    yn = pd.to_numeric(y, errors="coerce")
    return (yn > 0).astype(float).where(yn.notna(), np.nan)


def preprocess_dataset(df: pd.DataFrame):
    """Preprocessing aman untuk dataset NASA MDP dan dataset upload umum."""
    if df is None or df.empty:
        raise ValueError("Dataset kosong atau tidak terbaca.")

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    if df.shape[1] < 2:
        raise ValueError("Dataset harus memiliki minimal 1 fitur dan 1 target.")

    target_col = detect_target_column(df)

    # Drop kolom identitas/non-metrik jika ada
    drop_candidates = ["id", "ID", "name", "Name", "module", "Module", "version", "Version"]
    for drop_c in drop_candidates:
        if drop_c in df.columns and drop_c != target_col:
            df.drop(columns=[drop_c], inplace=True)

    y = normalize_target(df[target_col])

    X = df.drop(columns=[target_col])
    X = decode_bytes_dataframe(X)

    # Ubah fitur kategorikal menjadi numerik ringan, bukan langsung dibuang semua
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            num = pd.to_numeric(X[col], errors="coerce")
            # Kalau mayoritas bisa jadi numerik, pakai numerik; sisanya di-encode
            if num.notna().mean() >= 0.8:
                X[col] = num
            else:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("Tidak ada fitur numerik yang bisa digunakan setelah preprocessing.")

    # Buang baris target kosong, lalu isi missing value fitur dengan median
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int).to_numpy()

    if X.shape[0] == 0:
        raise ValueError("Tidak ada baris valid setelah target kosong dibuang.")

    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)

    # Buang fitur konstan karena tidak membantu FS/model
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    if X.shape[1] == 0:
        raise ValueError("Semua fitur konstan setelah preprocessing.")

    if len(np.unique(y)) < 2:
        raise ValueError("Target hanya memiliki 1 kelas unik setelah preprocessing.")

    return X, y, target_col


def safe_defect_ratio(defect_count: int, total_count: int) -> str:
    if total_count <= 0:
        return "0.00%"
    return f"{defect_count / total_count * 100:.2f}%"


def apply_feature_selection(method, X_train, y_train, X_test, k=None):
    if X_train.shape[1] == 0:
        raise ValueError("Tidak ada fitur yang tersedia untuk feature selection.")

    if method == "none":
        return X_train.values, X_test.values, list(X_train.columns)

    k = int(k or X_train.shape[1])
    k = max(1, min(k, X_train.shape[1]))

    if method == "chi2":
        sc = MinMaxScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)
        sel = SelectKBest(chi2, k=k)
    elif method == "mutual_info":
        Xtr, Xte = X_train, X_test
        sel = SelectKBest(mutual_info_classif, k=k)
    elif method == "anova":
        Xtr, Xte = X_train, X_test
        sel = SelectKBest(f_classif, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")

    Xtr_s = sel.fit_transform(Xtr, y_train)
    Xte_s = sel.transform(Xte)
    feat = list(X_train.columns[sel.get_support()])
    return Xtr_s, Xte_s, feat


def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    try:
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = np.nan
        y_proba = y_pred.astype(float)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "auc":       auc,
        "mcc":       matthews_corrcoef(y_test, y_pred),
    }
    return metrics, y_pred, y_proba


def to_excel_bytes(results_df: pd.DataFrame, summary_df: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if summary_df is not None:
            summary_df.to_excel(writer, sheet_name="Dataset Summary", index=False)

        results_df.to_excel(writer, sheet_name="All Results", index=False)

        best_ds = (
            results_df.sort_values(["dataset", "f1_score", "auc", "mcc"],
                                   ascending=[True, False, False, False])
            .groupby("dataset").head(1).reset_index(drop=True)
        )
        best_ds.to_excel(writer, sheet_name="Best Per Dataset", index=False)

        results_df.sort_values(["f1_score", "auc", "mcc"], ascending=False).head(10)\
            .to_excel(writer, sheet_name="Top 10 Overall", index=False)

        for agg_col, sheet in [("model", "Summary by Model"),
                                 ("feature_selection", "Summary by FS")]:
            grp = results_df.groupby(agg_col)[
                ["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
            ].mean().reset_index().round(4)
            grp.to_excel(writer, sheet_name=sheet, index=False)

        fs_only = results_df[results_df["feature_selection"] != "No Feature Selection"]
        if not fs_only.empty:
            grp_k = fs_only.groupby("k_requested")[
                ["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
            ].mean().reset_index().round(4)
            grp_k.to_excel(writer, sheet_name="Summary by K", index=False)

    return buf.getvalue()


# ─────────────────────────────────────────────
# AUTO LOAD DATASET DARI FOLDER data/
# ─────────────────────────────────────────────
# Dataset di streamlit/data/ akan otomatis dibaca saat app pertama kali dibuka.
load_datasets_from_data_folder(force=False)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Software Defect Prediction")
st.sidebar.caption("NASA MDP | Feature Selection | ML")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda", "📂 Dataset", "⚙️ Eksperimen", "📊 Hasil & Analisis"],
    label_visibility="collapsed",
)

# Status dataset & hasil di sidebar
n_ds = len(st.session_state.datasets)
n_res = len(st.session_state.results_df) if st.session_state.results_df is not None else 0
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset loaded:** {n_ds}")
st.sidebar.markdown(f"**Hasil eksperimen:** {n_res} skenario")
st.sidebar.caption(f"Data folder: `{get_data_dir()}`")


# ══════════════════════════════════════════════════════════
#  PAGE 1 – BERANDA
# ══════════════════════════════════════════════════════════
if page == "🏠 Beranda":
    st.markdown('<div class="main-title">🔬 Software Defect Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Comparative Analysis of Feature Selection Methods '
        'and Machine Learning Algorithms · NASA MDP Datasets</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.info("**📂 5 Dataset NASA MDP**\nJM1 · KC1 · PC1 · CM1 · KC3")
    c2.success("**⚙️ 4 Metode Feature Selection**\nChi-Square · MI · ANOVA · No FS")
    c3.warning("**🤖 3 Classifier**\nRandom Forest · SVM · XGBoost")

    st.markdown("---")

    col_info, col_result = st.columns(2)

    with col_info:
        st.markdown('<div class="sec-header">📋 Tentang Penelitian</div>', unsafe_allow_html=True)
        st.markdown("""
Penelitian ini membandingkan **metode seleksi fitur** dan **algoritma machine learning**
untuk prediksi cacat perangkat lunak menggunakan dataset NASA MDP.

| Komponen | Detail |
|---|---|
| Dataset | JM1, KC1, PC1, CM1, KC3 |
| Feature Selection | Chi-Square, Mutual Information, ANOVA F-test, No FS |
| Variasi k | 5, 10, 15, 20 |
| Algoritma | Random Forest, SVM, XGBoost |
| Metrik Evaluasi | Accuracy, Precision, Recall, F1-score, AUC, MCC |
| Total Skenario | **39 skenario** per dataset |

Evaluasi menekankan **F1-score, Recall, AUC, dan MCC** karena dataset tidak seimbang.
        """)

    with col_result:
        st.markdown('<div class="sec-header">🏆 Temuan Utama (dari Paper)</div>', unsafe_allow_html=True)
        st.markdown("**Rata-rata per Algoritma:**")
        tbl_model = pd.DataFrame({
            "Algoritma":   ["SVM ★", "XGBoost", "Random Forest"],
            "Accuracy":    [0.7276, 0.8179, 0.8165],
            "Recall":      [0.5658, 0.1641, 0.1379],
            "F1-Score":    [0.4021, 0.2239, 0.1842],
            "MCC":         [0.2684, 0.1608, 0.1215],
        })
        st.dataframe(tbl_model, hide_index=True, use_container_width=True)

        st.markdown("**Rata-rata per Feature Selection:**")
        tbl_fs = pd.DataFrame({
            "Feature Selection": ["ANOVA F-test ★", "Chi-Square", "No FS", "Mutual Info"],
            "F1-Score": [0.3002, 0.2772, 0.2325, 0.2422],
            "MCC":      [0.2146, 0.1850, 0.1602, 0.1570],
        })
        st.dataframe(tbl_fs, hide_index=True, use_container_width=True)

        st.markdown(
            '<div class="best-box">🏆 <b>Kombinasi Terbaik:</b> '
            'ANOVA F-test + k=20 + SVM<br>'
            'F1-Score = <b>0.4352</b> | MCC = <b>0.3149</b> | Recall = <b>0.5828</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sec-header">🚀 Cara Menggunakan</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown("**1️⃣ Dataset**\nSimpan file di folder `data/` agar auto-load, atau upload manual di 📂 Dataset.")
    s2.markdown("**2️⃣ Konfigurasi**\nBuka ⚙️ Eksperimen, pilih metode FS, nilai k, dan algoritma yang ingin diuji.")
    s3.markdown("**3️⃣ Jalankan**\nKlik tombol **Mulai Eksperimen** dan tunggu proses selesai.")
    s4.markdown("**4️⃣ Analisis**\nLihat hasil lengkap di 📊 Hasil & Analisis, dan download Excel.")


# ══════════════════════════════════════════════════════════
#  PAGE 2 – DATASET
# ══════════════════════════════════════════════════════════
elif page == "📂 Dataset":
    st.markdown('<div class="main-title">📂 Manajemen Dataset</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-header">Dataset Otomatis dari Folder data/</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Jika tidak ingin upload dataset terus, simpan file dataset di folder '
        '<b>data/</b> yang sejajar dengan <b>app.py</b>. Saat app dibuka, file <b>.arff</b>, '
        '<b>.csv</b>, <b>.xlsx</b>, atau <b>.xls</b> akan otomatis dibaca dulu.</div>',
        unsafe_allow_html=True,
    )

    data_dir = get_data_dir()
    if st.session_state.auto_loaded_files:
        st.success(f"✅ Dataset dari folder data sudah dicek: {data_dir}")
        st.dataframe(pd.DataFrame(st.session_state.auto_loaded_files), hide_index=True, use_container_width=True)
    else:
        st.info(f"Folder data yang dicek: `{data_dir}`. Jika belum ada, buat folder `data/` lalu masukkan dataset di sana.")

    c_load1, c_load2 = st.columns([1, 1])
    with c_load1:
        if st.button("🔄 Muat ulang dari folder data", use_container_width=True):
            st.session_state.autoload_done = False
            load_datasets_from_data_folder(force=True)
            st.session_state.results_df = None
            st.session_state.summary_df = None
            st.rerun()
    with c_load2:
        if st.button("🧹 Reset Dataset", use_container_width=True):
            st.session_state.datasets = {}
            st.session_state.results_df = None
            st.session_state.summary_df = None
            st.session_state.auto_loaded_files = []
            st.session_state.autoload_done = True
            st.rerun()

    st.markdown("---")
    st.markdown('<div class="sec-header">Upload Dataset Manual</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Bagian ini opsional. Upload manual dipakai jika ingin menambah dataset baru '
        'atau mengganti dataset dari folder <b>data/</b> tanpa mengubah file project.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    up_files = st.file_uploader(
        "Upload dataset sekaligus",
        type=["arff", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Bisa upload satu file atau banyak file dataset sekaligus.",
    )

    if up_files:
        loaded_rows = []
        for up in up_files:
            try:
                df = read_uploaded_file(up)
                if df is not None:
                    ds_name = infer_dataset_name(up.name, st.session_state.datasets.keys())
                    st.session_state.datasets[ds_name] = df
                    loaded_rows.append({
                        "Dataset": ds_name,
                        "File": up.name,
                        "Baris": df.shape[0],
                        "Kolom": df.shape[1],
                        "Status": "Berhasil dibaca",
                    })
            except Exception as e:
                loaded_rows.append({
                    "Dataset": "-",
                    "File": up.name,
                    "Baris": 0,
                    "Kolom": 0,
                    "Status": f"Gagal: {e}",
                })

        if loaded_rows:
            st.markdown("**Status upload:**")
            st.dataframe(pd.DataFrame(loaded_rows), hide_index=True, use_container_width=True)

    if st.session_state.datasets:
        st.markdown("**Dataset yang sedang aktif:**")
        active_df = pd.DataFrame([
            {"Dataset": name, "Jumlah Baris": df.shape[0], "Jumlah Kolom": df.shape[1]}
            for name, df in st.session_state.datasets.items()
        ])
        st.dataframe(active_df, hide_index=True, use_container_width=True)

    if not st.session_state.datasets:
        st.warning("⚠️ Belum ada dataset. Simpan file di folder data/ atau upload manual minimal satu dataset untuk melanjutkan.")
        st.stop()

    # ── Dataset Summary ────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-header">📊 Ringkasan Dataset</div>', unsafe_allow_html=True)

    summary_rows = []
    processed = {}
    for dsn, df in st.session_state.datasets.items():
        try:
            X, y, tc = preprocess_dataset(df)
            processed[dsn] = (X, y, tc)
            cd = pd.Series(y).value_counts().to_dict()
            nd, d = cd.get(0, 0), cd.get(1, 0)
            summary_rows.append({
                "Dataset": dsn, "Jumlah Data": X.shape[0],
                "Jumlah Fitur": X.shape[1], "Non-Defect": nd,
                "Defect": d, "Defect Ratio (%)": safe_defect_ratio(d, X.shape[0]),
            })
        except Exception as e:
            st.error(f"Error saat preprocessing {dsn}: {e}")

    if summary_rows:
        s_df = pd.DataFrame(summary_rows)
        st.session_state.summary_df = s_df
        st.dataframe(s_df, hide_index=True, use_container_width=True)

        # Pie charts
        st.markdown('<div class="sec-header">📈 Distribusi Kelas</div>', unsafe_allow_html=True)
        n_plots = len(summary_rows)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        PAL = ["#27ae60", "#e74c3c"]
        for ax, row in zip(axes, summary_rows):
            vals = [row["Non-Defect"], row["Defect"]]
            lbls = [f"Non-Defect\n({row['Non-Defect']:,})", f"Defect\n({row['Defect']:,})"]
            ax.pie(vals, labels=lbls, colors=PAL, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 9})
            ax.set_title(row["Dataset"], fontweight="bold", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Bar – class imbalance comparison
        st.markdown('<div class="sec-header">📉 Perbandingan Defect Ratio</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ratios = [float(r["Defect Ratio (%)"].replace("%", "")) for r in summary_rows]
        colors_bar = ["#e74c3c" if r < 15 else "#f39c12" if r < 25 else "#27ae60" for r in ratios]
        bars = ax2.bar([r["Dataset"] for r in summary_rows], ratios, color=colors_bar, edgecolor="white", linewidth=1.5)
        ax2.set_ylabel("Defect Ratio (%)", fontsize=11)
        ax2.set_title("Defect Ratio per Dataset", fontsize=13, fontweight="bold")
        ax2.axhline(20, color="gray", linestyle="--", linewidth=1, label="20% threshold")
        ax2.legend(fontsize=10)
        for bar, v in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax2.set_ylim(0, max(max(ratios) * 1.3, 1))
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Individual Exploration ─────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-header">🔍 Eksplorasi Dataset Individual</div>', unsafe_allow_html=True)
    if not processed:
        st.warning("Tidak ada dataset valid yang bisa dieksplorasi. Cek kolom target dan isi dataset.")
        st.stop()

    sel_ds = st.selectbox("Pilih Dataset:", list(processed.keys()))
    if sel_ds:
        X_e, y_e, tc_e = processed[sel_ds]
        t1, t2, t3, t4 = st.tabs(["📋 Preview Data", "📈 Statistik", "📊 Distribusi Fitur", "🔥 Korelasi"])

        with t1:
            st.caption(f"Target column: **{tc_e}** | Shape: {X_e.shape}")
            st.dataframe(X_e.head(20), use_container_width=True)

        with t2:
            st.dataframe(X_e.describe().round(4), use_container_width=True)

        with t3:
            feat_opts = list(X_e.columns)
            chosen = st.selectbox("Pilih Fitur:", feat_opts)
            if chosen:
                fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
                # histogram
                c0 = X_e[y_e == 0][chosen].dropna()
                c1 = X_e[y_e == 1][chosen].dropna()
                axes3[0].hist(c0, bins=30, alpha=0.6, color="#27ae60", label="Non-Defect")
                axes3[0].hist(c1, bins=30, alpha=0.6, color="#e74c3c", label="Defect")
                axes3[0].set_title(f"Distribusi '{chosen}' per Kelas", fontsize=11)
                axes3[0].legend()
                axes3[0].set_xlabel(chosen)
                axes3[0].set_ylabel("Frekuensi")
                # boxplot
                data_box = [c0.values, c1.values]
                bp = axes3[1].boxplot(data_box, patch_artist=True,
                                      labels=["Non-Defect", "Defect"],
                                      boxprops=dict(facecolor="#ddeeff"))
                bp["boxes"][0].set_facecolor("#27ae60")
                bp["boxes"][1].set_facecolor("#e74c3c")
                axes3[1].set_title(f"Boxplot '{chosen}'", fontsize=11)
                axes3[1].set_ylabel(chosen)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()

        with t4:
            max_feat = min(X_e.shape[1], 25)
            fig4, ax4 = plt.subplots(figsize=(12, 10))
            corr = X_e.iloc[:, :max_feat].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=(max_feat <= 15), fmt=".2f",
                        cmap="coolwarm", center=0, ax=ax4,
                        linewidths=0.3, cbar_kws={"shrink": .8})
            ax4.set_title(f"Heatmap Korelasi – {sel_ds} (top {max_feat} fitur)", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
            if X_e.shape[1] > 25:
                st.caption("ℹ️ Hanya 25 fitur pertama yang ditampilkan.")


# ══════════════════════════════════════════════════════════
#  PAGE 3 – EKSPERIMEN
# ══════════════════════════════════════════════════════════
elif page == "⚙️ Eksperimen":
    st.markdown('<div class="main-title">⚙️ Konfigurasi & Jalankan Eksperimen</div>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.error("❌ Belum ada dataset. Upload dataset di halaman 📂 Dataset terlebih dahulu.")
        st.stop()

    col_cfg, col_sum = st.columns([1, 2])

    # ── Konfigurasi ────────────────────────────────────────
    with col_cfg:
        st.markdown('<div class="sec-header">🎛️ Konfigurasi</div>', unsafe_allow_html=True)

        sel_ds_list = st.multiselect(
            "Dataset yang digunakan:",
            list(st.session_state.datasets.keys()),
            default=list(st.session_state.datasets.keys()),
        )

        st.markdown("**Metode Feature Selection:**")
        chk_nofs = st.checkbox("No Feature Selection (Baseline)", value=True)
        chk_chi2 = st.checkbox("Chi-Square", value=True)
        chk_mi   = st.checkbox("Mutual Information", value=True)
        chk_anova = st.checkbox("ANOVA F-test", value=True)

        k_vals = st.multiselect("Nilai k (fitur terpilih):", [5, 10, 15, 20], default=[5, 10, 15, 20])

        st.markdown("**Algoritma ML:**")
        chk_rf  = st.checkbox("Random Forest", value=True)
        chk_svm = st.checkbox("SVM", value=True)
        chk_xgb = st.checkbox("XGBoost", value=True)

        test_size = st.slider("Test size (%)", 10, 40, 20, 5) / 100

    # ── Ringkasan ──────────────────────────────────────────
    with col_sum:
        st.markdown('<div class="sec-header">📋 Ringkasan Konfigurasi</div>', unsafe_allow_html=True)

        fs_map = {}
        if chk_nofs:  fs_map["No Feature Selection"] = "none"
        if chk_chi2:  fs_map["Chi-Square"]           = "chi2"
        if chk_mi:    fs_map["Mutual Information"]    = "mutual_info"
        if chk_anova: fs_map["ANOVA F-test"]          = "anova"

        sel_models = (
            (["Random Forest"] if chk_rf else []) +
            (["SVM"]           if chk_svm else []) +
            (["XGBoost"]       if chk_xgb else [])
        )

        n_with_k  = sum(1 for v in fs_map.values() if v != "none")
        n_no_k    = sum(1 for v in fs_map.values() if v == "none")
        total_scen = len(sel_ds_list) * (n_with_k * len(k_vals) + n_no_k) * len(sel_models)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Dataset", len(sel_ds_list))
        r2.metric("Metode FS", len(fs_map))
        r3.metric("Algoritma", len(sel_models))
        r4.metric("Total Skenario", total_scen)

        st.markdown("---")
        st.write("**Dataset:**", ", ".join(sel_ds_list) or "—")
        st.write("**Feature Selection:**", ", ".join(fs_map.keys()) or "—")
        st.write("**k values:**", str(k_vals) if k_vals else "—")
        st.write("**Algoritma:**", ", ".join(sel_models) or "—")
        st.write(f"**Train / Test split:** {int((1-test_size)*100)}% / {int(test_size*100)}%")

        st.markdown(
            '<div class="warn-box">⚠️ <b>Perhatian:</b> SVM pada dataset besar (JM1) '
            'butuh waktu relatif lama. Pastikan semua konfigurasi sudah benar sebelum '
            'menjalankan eksperimen.</div>',
            unsafe_allow_html=True,
        )

    # ── Validation & Run ───────────────────────────────────
    st.markdown("---")
    err_msgs = []
    if not sel_ds_list:    err_msgs.append("Pilih minimal satu dataset.")
    if not fs_map:         err_msgs.append("Pilih minimal satu metode feature selection.")
    if not sel_models:     err_msgs.append("Pilih minimal satu algoritma.")
    if not k_vals and any(v != "none" for v in fs_map.values()):
        err_msgs.append("Pilih minimal satu nilai k.")
    for msg in err_msgs:
        st.error(f"❌ {msg}")

    run_btn = st.button(
        "🚀 Mulai Eksperimen", type="primary", use_container_width=True,
        disabled=bool(err_msgs),
    )

    if run_btn:
        results   = []
        progress  = st.progress(0)
        status    = st.empty()
        completed = 0

        log_expander = st.expander("📜 Log Eksperimen (real-time)", expanded=True)
        log_area     = log_expander.empty()
        log_lines    = []

        def push_log(msg):
            log_lines.append(msg)
            log_area.text_area("", "\n".join(log_lines[-80:]), height=250, key=f"log_{len(log_lines)}")

        for dsn in sel_ds_list:
            df = st.session_state.datasets[dsn]
            try:
                X, y, tc = preprocess_dataset(df)
            except Exception as e:
                push_log(f"[ERROR] {dsn}: {e}")
                continue

            if X.shape[0] == 0:
                push_log(f"[SKIP] {dsn}: data kosong setelah preprocessing.")
                continue

            class_counts = pd.Series(y).value_counts().to_dict()
            if len(class_counts) < 2:
                push_log(f"[SKIP] {dsn}: hanya 1 kelas unik setelah preprocessing.")
                continue
            if min(class_counts.values()) < 2:
                push_log(f"[SKIP] {dsn}: jumlah sampel pada salah satu kelas kurang dari 2, tidak aman untuk stratified split.")
                continue

            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except Exception as e:
                push_log(f"[SPLIT-ERR] {dsn}: {e}")
                continue

            push_log(f"\n{'='*60}")
            push_log(f"[DATASET] {dsn} | {X.shape[0]} data | {X.shape[1]} fitur | "
                     f"Defect: {y.sum()} ({y.mean()*100:.1f}%)")

            for fs_name, fs_method in fs_map.items():
                k_loop = ["all"] if fs_method == "none" else k_vals

                for k in k_loop:
                    try:
                        k_arg = None if fs_method == "none" else k
                        Xtr_fs, Xte_fs, sel_feat = apply_feature_selection(
                            fs_method, X_tr, y_tr, X_te, k=k_arg
                        )
                        actual_k = len(sel_feat)
                    except Exception as e:
                        push_log(f"  [FS-ERR] {fs_name} k={k}: {e}")
                        completed += len(sel_models)
                        progress.progress(min(completed / max(total_scen, 1), 1.0))
                        continue

                    for mdl_name in sel_models:
                        status.markdown(
                            f"⏳ **{dsn}** | {fs_name} | k={k} | **{mdl_name}**"
                        )
                        model = get_models()[mdl_name]
                        Xtr_m, Xte_m = Xtr_fs.copy(), Xte_fs.copy()
                        if mdl_name == "SVM":
                            sc = StandardScaler()
                            Xtr_m = sc.fit_transform(Xtr_m)
                            Xte_m = sc.transform(Xte_m)
                        try:
                            t0 = time.time()
                            model.fit(Xtr_m, y_tr)
                            train_t = time.time() - t0
                            mets, y_pred, y_proba = evaluate_model(model, Xte_m, y_te)
                            cm = confusion_matrix(y_te, y_pred)
                            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (np.nan,) * 4
                            results.append({
                                "dataset": dsn, "feature_selection": fs_name,
                                "k_requested": k, "k_actual": actual_k,
                                "model": mdl_name,
                                "accuracy":  round(mets["accuracy"],  4),
                                "precision": round(mets["precision"], 4),
                                "recall":    round(mets["recall"],    4),
                                "f1_score":  round(mets["f1_score"],  4),
                                "auc":       round(mets["auc"],       4),
                                "mcc":       round(mets["mcc"],       4),
                                "training_time_s": round(train_t, 3),
                                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                            })
                            push_log(
                                f"  ✅ {fs_name} | k={k} | {mdl_name} → "
                                f"Acc:{mets['accuracy']:.4f} Prec:{mets['precision']:.4f} "
                                f"Rec:{mets['recall']:.4f} F1:{mets['f1_score']:.4f} "
                                f"AUC:{mets['auc']:.4f} MCC:{mets['mcc']:.4f} "
                                f"({train_t:.1f}s)"
                            )
                        except Exception as e:
                            push_log(f"  [MODEL-ERR] {mdl_name}: {e}")

                        completed += 1
                        progress.progress(min(completed / max(total_scen, 1), 1.0))

        if results:
            rdf = pd.DataFrame(results)
            st.session_state.results_df = rdf
            progress.progress(1.0)
            status.success(f"✅ Eksperimen selesai! **{len(results)}** skenario berhasil.")

            best = rdf.sort_values(["f1_score", "mcc"], ascending=False).head(5)
            st.markdown('<div class="sec-header">🏆 Top 5 Kombinasi Terbaik</div>', unsafe_allow_html=True)
            st.dataframe(
                best[["dataset", "feature_selection", "k_requested", "model",
                       "accuracy", "precision", "recall", "f1_score", "auc", "mcc"]]
                .reset_index(drop=True),
                hide_index=True, use_container_width=True,
            )
        else:
            status.error("❌ Tidak ada hasil. Periksa konfigurasi dan dataset.")


# ══════════════════════════════════════════════════════════
#  PAGE 4 – HASIL & ANALISIS
# ══════════════════════════════════════════════════════════
elif page == "📊 Hasil & Analisis":
    st.markdown('<div class="main-title">📊 Hasil & Analisis</div>', unsafe_allow_html=True)

    if st.session_state.results_df is None:
        st.warning("⚠️ Belum ada hasil eksperimen. Jalankan eksperimen di halaman ⚙️ Eksperimen.")
        st.stop()

    rdf = st.session_state.results_df.copy()

    # ── Filter panel ───────────────────────────────────────
    with st.expander("🔍 Filter & Sorting", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        f_ds  = fc1.multiselect("Dataset:",          rdf["dataset"].unique().tolist(),          default=rdf["dataset"].unique().tolist())
        f_fs  = fc2.multiselect("Feature Selection:", rdf["feature_selection"].unique().tolist(), default=rdf["feature_selection"].unique().tolist())
        f_mdl = fc3.multiselect("Model:",             rdf["model"].unique().tolist(),             default=rdf["model"].unique().tolist())
        sort_m = fc4.selectbox("Urutkan berdasarkan:", ["f1_score", "recall", "auc", "mcc", "accuracy", "precision"])

    fdf = rdf[
        rdf["dataset"].isin(f_ds) &
        rdf["feature_selection"].isin(f_fs) &
        rdf["model"].isin(f_mdl)
    ]

    if fdf.empty:
        st.warning("Tidak ada data yang cocok dengan filter.")
        st.stop()

    # ── Tabs ───────────────────────────────────────────────
    t_all, t_mdl, t_fs, t_k, t_ds, t_best = st.tabs([
        "📋 Semua Hasil",
        "🤖 Per Algoritma",
        "⚙️ Per Feature Selection",
        "📊 Per Nilai k",
        "🗂️ Per Dataset",
        "🏆 Perbandingan & Terbaik",
    ])

    # ── Tab 1: ALL ─────────────────────────────────────────
    with t_all:
        st.markdown('<div class="sec-header">Semua Hasil Eksperimen</div>', unsafe_allow_html=True)
        show_cols = ["dataset", "feature_selection", "k_requested", "model",
                     "accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
        disp = fdf[show_cols].sort_values(sort_m, ascending=False).reset_index(drop=True)
        st.dataframe(
            disp.style.background_gradient(subset=["f1_score", "recall", "mcc", "auc"], cmap="YlGn"),
            use_container_width=True, height=420,
        )
        xl = to_excel_bytes(fdf, st.session_state.summary_df)
        st.download_button(
            "📥 Download Hasil (Excel)", data=xl,
            file_name="hasil_eksperimen_defect_prediction.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── Tab 2: PER ALGORITMA ───────────────────────────────
    with t_mdl:
        st.markdown('<div class="sec-header">Perbandingan Algoritma ML</div>', unsafe_allow_html=True)
        sum_mdl = (
            fdf.groupby("model")[["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]]
            .mean().reset_index().round(4)
            .sort_values("f1_score", ascending=False)
        )
        st.dataframe(sum_mdl, hide_index=True, use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        METRICS = ["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
        x = np.arange(len(METRICS))
        W = 0.25
        COLORS = ["#3498db", "#e74c3c", "#2ecc71"]
        for i, (_, row) in enumerate(sum_mdl.iterrows()):
            vals = [row[m] for m in METRICS]
            ax.bar(x + i * W, vals, W, label=row["model"],
                   color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="white")
        ax.set_xticks(x + W)
        ax.set_xticklabels([m.replace("_", "-").upper() for m in METRICS], rotation=10, fontsize=10)
        ax.set_ylabel("Nilai Rata-rata", fontsize=11)
        ax.set_title("Rata-rata Performa per Algoritma ML", fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ROC curve (per dataset)
        st.markdown("#### 📈 Distribusi F1-Score per Dataset × Algoritma")
        pivot_f1 = fdf.pivot_table(values="f1_score", index="dataset",
                                    columns="model", aggfunc="mean").round(4)
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        pivot_f1.plot(kind="bar", ax=ax2, color=COLORS[:len(pivot_f1.columns)],
                      alpha=0.85, edgecolor="white")
        ax2.set_ylabel("Rata-rata F1-Score", fontsize=11)
        ax2.set_title("F1-Score per Dataset × Algoritma", fontsize=13, fontweight="bold")
        ax2.legend(title="Model", fontsize=10)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=11)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Tab 3: PER FEATURE SELECTION ──────────────────────
    with t_fs:
        st.markdown('<div class="sec-header">Perbandingan Metode Feature Selection</div>', unsafe_allow_html=True)
        sum_fs = (
            fdf.groupby("feature_selection")[["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]]
            .mean().reset_index().round(4)
            .sort_values("f1_score", ascending=False)
        )
        st.dataframe(sum_fs, hide_index=True, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        C_FS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        for ax, metric in zip(axes, ["f1_score", "recall", "mcc"]):
            bars = ax.barh(sum_fs["feature_selection"], sum_fs[metric],
                           color=C_FS[:len(sum_fs)], alpha=0.85, edgecolor="white")
            ax.set_xlabel(f"Rata-rata {metric.replace('_', '-').upper()}", fontsize=10)
            ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
            for bar, v in zip(bars, sum_fs[metric]):
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=9)
            ax.set_xlim(0, max(sum_fs[metric]) * 1.25)
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: PER K ────────────────────────────────────────
    with t_k:
        st.markdown('<div class="sec-header">Pengaruh Jumlah Fitur yang Dipilih (k)</div>', unsafe_allow_html=True)
        fs_only = fdf[fdf["feature_selection"] != "No Feature Selection"]
        if fs_only.empty:
            st.info("Tidak ada data feature selection untuk ditampilkan.")
        else:
            sum_k = (
                fs_only.groupby("k_requested")[["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]]
                .mean().reset_index().round(4)
                .sort_values("k_requested")
            )
            st.dataframe(sum_k, hide_index=True, use_container_width=True)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            LINE_COLORS = {"recall": "#e74c3c", "f1_score": "#3498db",
                           "auc": "#2ecc71", "mcc": "#9b59b6"}
            for met, col in LINE_COLORS.items():
                ax.plot(sum_k["k_requested"], sum_k[met], marker="o",
                        linewidth=2.5, label=met.replace("_", "-").upper(),
                        color=col, markersize=8)
                for x_, y_ in zip(sum_k["k_requested"], sum_k[met]):
                    ax.annotate(f"{y_:.3f}", (x_, y_),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8, color=col)
            ax.set_xlabel("Jumlah Fitur Terpilih (k)", fontsize=12)
            ax.set_ylabel("Nilai Metrik (rata-rata)", fontsize=12)
            ax.set_title("Pengaruh k terhadap Performa Prediksi Cacat", fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.set_xticks(sum_k["k_requested"])
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Per FS
            st.markdown("#### Performa per k × Feature Selection")
            pivot_k_fs = fs_only.pivot_table(values="f1_score", index="k_requested",
                                              columns="feature_selection", aggfunc="mean").round(4)
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            for col_name in pivot_k_fs.columns:
                ax2.plot(pivot_k_fs.index, pivot_k_fs[col_name], marker="s",
                         linewidth=2, label=col_name, markersize=7)
            ax2.set_xlabel("k", fontsize=12)
            ax2.set_ylabel("Rata-rata F1-Score", fontsize=12)
            ax2.set_title("F1-Score per k × Metode FS", fontsize=13, fontweight="bold")
            ax2.legend(fontsize=10)
            ax2.set_xticks(pivot_k_fs.index)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── Tab 5: PER DATASET ─────────────────────────────────
    with t_ds:
        st.markdown('<div class="sec-header">Analisis per Dataset</div>', unsafe_allow_html=True)
        ds_choice = st.selectbox("Pilih Dataset:", fdf["dataset"].unique().tolist(), key="ds_tab5")
        ds_df = fdf[fdf["dataset"] == ds_choice]

        st.markdown(f"#### Best 5 Skenario – {ds_choice}")
        best5 = ds_df.sort_values(["f1_score", "mcc"], ascending=False).head(5)[
            ["feature_selection", "k_requested", "model",
             "accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
        ].reset_index(drop=True)
        best5.index += 1
        st.dataframe(best5, use_container_width=True)

        st.markdown(f"#### Heatmap F1-Score – {ds_choice}")
        piv = ds_df.pivot_table(values="f1_score", index="feature_selection",
                                 columns="model", aggfunc="mean").round(4)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(piv, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "F1-Score"})
        ax.set_title(f"F1-Score Heatmap – {ds_choice}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion matrix best model
        st.markdown(f"#### Confusion Matrix – Skenario Terbaik ({ds_choice})")
        best_row = ds_df.sort_values(["f1_score", "mcc"], ascending=False).iloc[0]
        if not any(np.isnan([best_row["tp"], best_row["tn"], best_row["fp"], best_row["fn"]])):
            cm_arr = np.array([[int(best_row["tn"]), int(best_row["fp"])],
                               [int(best_row["fn"]), int(best_row["tp"])]])
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax2,
                        xticklabels=["Non-Defect", "Defect"],
                        yticklabels=["Non-Defect", "Defect"],
                        linewidths=1, linecolor="white")
            ax2.set_xlabel("Predicted", fontsize=11)
            ax2.set_ylabel("Actual", fontsize=11)
            ax2.set_title(
                f"{best_row['feature_selection']} k={best_row['k_requested']} {best_row['model']}\n"
                f"F1={best_row['f1_score']:.4f} MCC={best_row['mcc']:.4f}",
                fontsize=10,
            )
            plt.tight_layout()
            col_cm, _ = st.columns([1, 1])
            col_cm.pyplot(fig2)
            plt.close()
        else:
            st.info("Confusion matrix tidak tersedia untuk skenario ini.")

    # ── Tab 6: PERBANDINGAN & TERBAIK ─────────────────────
    with t_best:
        st.markdown('<div class="sec-header">Perbandingan Menyeluruh & Kombinasi Terbaik</div>', unsafe_allow_html=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 🥇 Top 10 Kombinasi (F1-Score)")
            top10 = (
                fdf.sort_values(["f1_score", "mcc"], ascending=False).head(10)
                [["feature_selection", "k_requested", "model",
                  "f1_score", "recall", "auc", "mcc"]]
                .reset_index(drop=True)
            )
            top10.index += 1
            st.dataframe(
                top10.style.background_gradient(subset=["f1_score", "mcc"], cmap="YlGn"),
                use_container_width=True,
            )

        with c_right:
            st.markdown("#### 📌 Best per Dataset")
            bpd = (
                fdf.sort_values(["dataset", "f1_score", "mcc"], ascending=[True, False, False])
                .groupby("dataset").head(1).reset_index(drop=True)
                [["dataset", "feature_selection", "k_requested", "model",
                  "f1_score", "recall", "mcc"]]
            )
            st.dataframe(bpd, hide_index=True, use_container_width=True)

        # Heatmap FS × Model
        st.markdown("#### 🔥 Heatmap Rata-rata F1-Score: FS × Model")
        piv_all = fdf.pivot_table(values="f1_score", index="feature_selection",
                                   columns="model", aggfunc="mean").round(4)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(piv_all, annot=True, fmt=".4f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "F1-Score"})
        ax.set_title("Heatmap F1-Score: Feature Selection × Algoritma", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Radar / Spider chart
        st.markdown("#### 🕸️ Radar Chart – Algoritma (rata-rata semua skenario)")
        metrics_radar = ["accuracy", "precision", "recall", "f1_score", "auc", "mcc"]
        labels_radar  = [m.replace("_", "-").upper() for m in metrics_radar]
        N = len(labels_radar)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig3, ax3 = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        COLORS_R = ["#3498db", "#e74c3c", "#2ecc71"]
        sum_mdl_r = (
            fdf.groupby("model")[metrics_radar].mean().reset_index()
        )
        for i, (_, row) in enumerate(sum_mdl_r.iterrows()):
            vals = [row[m] for m in metrics_radar]
            vals += vals[:1]
            ax3.plot(angles, vals, linewidth=2, linestyle="solid",
                     label=row["model"], color=COLORS_R[i % len(COLORS_R)])
            ax3.fill(angles, vals, alpha=0.1, color=COLORS_R[i % len(COLORS_R)])
        ax3.set_thetagrids(np.degrees(angles[:-1]), labels_radar, fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.set_title("Radar Chart – Performa Algoritma", fontsize=13, fontweight="bold", y=1.1)
        ax3.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        st.pyplot(fig3)
        plt.close()

        # Best combination highlight
        best_overall = fdf.sort_values(["f1_score", "mcc"], ascending=False).iloc[0]
        st.markdown("---")
        st.markdown(
            f'<div class="best-box">'
            f'🏆 <b>Kombinasi Terbaik dari Eksperimen Ini</b><br><br>'
            f'<b>Dataset:</b> {best_overall["dataset"]} &nbsp;|&nbsp; '
            f'<b>Feature Selection:</b> {best_overall["feature_selection"]} &nbsp;|&nbsp; '
            f'<b>k:</b> {best_overall["k_requested"]} &nbsp;|&nbsp; '
            f'<b>Model:</b> {best_overall["model"]}<br><br>'
            f'F1-Score = <b>{best_overall["f1_score"]:.4f}</b> &nbsp;&nbsp; '
            f'Recall = <b>{best_overall["recall"]:.4f}</b> &nbsp;&nbsp; '
            f'AUC = <b>{best_overall["auc"]:.4f}</b> &nbsp;&nbsp; '
            f'MCC = <b>{best_overall["mcc"]:.4f}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )