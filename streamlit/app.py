# app.py

import io
import os
import time
import tempfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Software Defect Prediction Experiment")
st.write(
    """
    Aplikasi ini digunakan untuk menjalankan eksperimen penelitian 
    **Software Defect Prediction** menggunakan beberapa metode 
    **Feature Selection** dan algoritma **Machine Learning**.
    """
)


# =========================================================
# FUNGSI MEMBACA DATASET
# =========================================================

def read_dataset(uploaded_file):
    """
    Membaca dataset dari file upload.
    Format yang didukung: ARFF, CSV, XLSX, XLS.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df

    elif filename.endswith(".arff"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".arff") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            data, meta = arff.loadarff(tmp_path)
            df = pd.DataFrame(data)

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )

            return df

        finally:
            os.remove(tmp_path)

    else:
        raise ValueError("Format file tidak didukung. Gunakan ARFF, CSV, XLSX, atau XLS.")


# =========================================================
# FUNGSI PREPROCESSING DATASET
# =========================================================

def detect_target_column(df):
    """
    Mendeteksi kolom target secara otomatis.
    Jika tidak ditemukan, sistem akan memakai kolom terakhir.
    """

    possible_targets = [
        "defects",
        "Defective",
        "defective",
        "bug",
        "bugs",
        "class",
        "Class",
        "target",
        "Target",
        "label",
        "Label",
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    return df.columns[-1]


def encode_target(y):
    """
    Mengubah target menjadi 0 dan 1.
    """

    y = y.copy()

    y = y.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == bool:
        y = y.astype(str).str.strip().str.lower()

        mapping = {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "defect": 1,
            "defective": 1,
            "bug": 1,
            "buggy": 1,
            "faulty": 1,
            "clean": 0,
            "non-defect": 0,
            "non_defect": 0,
            "not_buggy": 0,
            "not faulty": 0,
            "0": 0,
            "1": 1,
        }

        if set(y.dropna().unique()).issubset(set(mapping.keys())):
            y = y.map(mapping)
        else:
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

    else:
        y = pd.to_numeric(y, errors="coerce")

    return pd.Series(y)


def preprocess_dataset(df, target_column=None):
    """
    Tahapan preprocessing:
    1. Merapikan nama kolom
    2. Menentukan kolom target
    3. Menghapus kolom ID/name/module jika ada
    4. Mengubah target menjadi 0/1
    5. Mengubah fitur menjadi numerik
    6. Mengisi missing value dengan median
    """

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if target_column and target_column in df.columns:
        target_col = target_column
    else:
        target_col = detect_target_column(df)

    possible_id_cols = ["id", "ID", "name", "Name", "module", "Module"]

    for col in possible_id_cols:
        if col in df.columns and col != target_col:
            df = df.drop(columns=[col])

    y = encode_target(df[target_col])

    X = df.drop(columns=[target_col])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(int).values

    return X, y, target_col


# =========================================================
# FUNGSI FEATURE SELECTION
# =========================================================

def apply_feature_selection(method, X_train, y_train, X_test, k=None):
    """
    Menerapkan feature selection:
    - none
    - chi2
    - mutual_info
    - anova
    """

    if method == "none":
        selected_features = list(X_train.columns)
        return X_train.values, X_test.values, selected_features

    if k is None:
        raise ValueError("Nilai k harus diisi.")

    k = min(k, X_train.shape[1])

    if method == "chi2":
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        selector = SelectKBest(score_func=chi2, k=k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

    elif method == "mutual_info":
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    elif method == "anova":
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

    else:
        raise ValueError("Metode feature selection tidak dikenal.")

    selected_features = list(X_train.columns[selector.get_support()])

    return X_train_selected, X_test_selected, selected_features


# =========================================================
# FUNGSI MODEL
# =========================================================

def get_models(selected_models):
    """
    Mengambil model yang dipilih user.
    """

    all_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        ),
    }

    return {name: all_models[name] for name in selected_models}


# =========================================================
# FUNGSI EVALUASI
# =========================================================

def evaluate_model(model, X_test, y_test):
    """
    Menghitung metrik evaluasi model.
    """

    y_pred = model.predict(X_test)

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        auc = roc_auc_score(y_test, y_proba)

    except Exception:
        auc = np.nan

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc": auc,
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    return metrics, y_pred


# =========================================================
# FUNGSI EXPORT EXCEL
# =========================================================

def create_excel_file(
    dataset_summary_df,
    results_df,
    best_by_dataset,
    best_overall,
    summary_by_model,
    summary_by_fs,
    summary_by_k,
    summary_by_combination,
    selected_features_df,
):
    """
    Membuat file Excel hasil eksperimen.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataset_summary_df.to_excel(writer, sheet_name="Dataset Summary", index=False)
        results_df.to_excel(writer, sheet_name="All Results", index=False)
        best_by_dataset.to_excel(writer, sheet_name="Best Per Dataset", index=False)
        best_overall.to_excel(writer, sheet_name="Best Overall", index=False)
        summary_by_model.to_excel(writer, sheet_name="Summary Model", index=False)
        summary_by_fs.to_excel(writer, sheet_name="Summary FS", index=False)

        if not summary_by_k.empty:
            summary_by_k.to_excel(writer, sheet_name="Summary K", index=False)

        summary_by_combination.to_excel(writer, sheet_name="Summary Combination", index=False)
        selected_features_df.to_excel(writer, sheet_name="Selected Features", index=False)

    output.seek(0)

    return output


# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("⚙️ Pengaturan Eksperimen")

uploaded_files = st.sidebar.file_uploader(
    "Upload dataset",
    type=["arff", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

target_column_input = st.sidebar.text_input(
    "Nama kolom target, opsional",
    value="",
    help="Kosongkan jika ingin target dideteksi otomatis.",
)

selected_fs = st.sidebar.multiselect(
    "Pilih Feature Selection",
    options=[
        "No Feature Selection",
        "Chi-Square",
        "Mutual Information",
        "ANOVA F-test",
    ],
    default=[
        "No Feature Selection",
        "Chi-Square",
        "Mutual Information",
        "ANOVA F-test",
    ],
)

selected_k = st.sidebar.multiselect(
    "Pilih nilai k",
    options=[5, 10, 15, 20],
    default=[5, 10, 15, 20],
)

selected_models = st.sidebar.multiselect(
    "Pilih Model",
    options=["Random Forest", "SVM", "XGBoost"],
    default=["Random Forest", "SVM", "XGBoost"],
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    value=42,
    step=1,
)

run_button = st.sidebar.button("🚀 Jalankan Eksperimen")


# =========================================================
# TAMPILAN AWAL
# =========================================================

st.subheader("📌 Alur Penelitian")

st.markdown(
    """
    Alur aplikasi:

    1. Upload dataset.
    2. Sistem melakukan preprocessing.
    3. Dataset dibagi menjadi data training dan testing.
    4. Sistem melakukan feature selection.
    5. Model dilatih menggunakan data training.
    6. Model diuji menggunakan data testing.
    7. Hasil evaluasi ditampilkan dan dapat diunduh dalam format Excel.
    """
)

if uploaded_files:
    st.subheader("📁 Preview Dataset")

    tabs = st.tabs([file.name for file in uploaded_files])

    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            try:
                df_preview = read_dataset(uploaded_file)

                st.write(
                    f"Ukuran dataset: **{df_preview.shape[0]} baris** × "
                    f"**{df_preview.shape[1]} kolom**"
                )

                detected_target = detect_target_column(df_preview)
                st.write(f"Kolom target terdeteksi: **{detected_target}**")

                st.dataframe(df_preview.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Gagal membaca dataset: {e}")

else:
    st.info("Silakan upload dataset terlebih dahulu melalui sidebar.")


# =========================================================
# PROSES EKSPERIMEN
# =========================================================

if run_button:

    if not uploaded_files:
        st.error("Dataset belum diupload.")
        st.stop()

    if not selected_fs:
        st.error("Pilih minimal satu metode feature selection.")
        st.stop()

    if not selected_models:
        st.error("Pilih minimal satu model.")
        st.stop()

    if any(fs != "No Feature Selection" for fs in selected_fs) and not selected_k:
        st.error("Pilih minimal satu nilai k.")
        st.stop()

    feature_methods = {
        "No Feature Selection": "none",
        "Chi-Square": "chi2",
        "Mutual Information": "mutual_info",
        "ANOVA F-test": "anova",
    }

    dataset_summary = []
    results = []
    selected_features_records = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_process = len(uploaded_files) * len(selected_fs) * len(selected_models)
    current_process = 0

    for uploaded_file in uploaded_files:

        dataset_name = os.path.splitext(uploaded_file.name)[0]

        status_text.info(f"Memproses dataset: {dataset_name}")

        df = read_dataset(uploaded_file)

        target_column = target_column_input.strip() if target_column_input.strip() else None

        X, y, target_col = preprocess_dataset(df, target_column)

        class_dist = pd.Series(y).value_counts().to_dict()

        dataset_summary.append({
            "dataset": dataset_name,
            "target_column": target_col,
            "num_data": X.shape[0],
            "num_features": X.shape[1],
            "class_0": class_dist.get(0, 0),
            "class_1": class_dist.get(1, 0),
            "defect_ratio": class_dist.get(1, 0) / X.shape[0],
        })

        if len(np.unique(y)) < 2:
            st.warning(f"Dataset {dataset_name} dilewati karena hanya memiliki satu kelas.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=int(random_state),
            stratify=y,
        )

        for fs_name in selected_fs:

            fs_method = feature_methods[fs_name]

            if fs_method == "none":
                k_values = ["all"]
            else:
                k_values = selected_k

            for k in k_values:

                if fs_method == "none":
                    X_train_fs, X_test_fs, selected_features = apply_feature_selection(
                        method=fs_method,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        k=None,
                    )

                    actual_k = X_train.shape[1]

                else:
                    X_train_fs, X_test_fs, selected_features = apply_feature_selection(
                        method=fs_method,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        k=int(k),
                    )

                    actual_k = len(selected_features)

                selected_features_records.append({
                    "dataset": dataset_name,
                    "feature_selection": fs_name,
                    "k_requested": k,
                    "k_actual": actual_k,
                    "selected_features": ", ".join(selected_features),
                })

                models = get_models(selected_models)

                for model_name, model in models.items():

                    status_text.info(
                        f"Dataset: {dataset_name} | FS: {fs_name} | "
                        f"k: {k} | Model: {model_name}"
                    )

                    X_train_model = X_train_fs
                    X_test_model = X_test_fs

                    if model_name == "SVM":
                        scaler = StandardScaler()
                        X_train_model = scaler.fit_transform(X_train_model)
                        X_test_model = scaler.transform(X_test_model)

                    start_time = time.time()
                    model.fit(X_train_model, y_train)
                    training_time = time.time() - start_time

                    metrics, y_pred = evaluate_model(model, X_test_model, y_test)

                    cm = confusion_matrix(y_test, y_pred)

                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

                    results.append({
                        "dataset": dataset_name,
                        "target_column": target_col,
                        "feature_selection": fs_name,
                        "k_requested": k,
                        "k_actual": actual_k,
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "auc": metrics["auc"],
                        "mcc": metrics["mcc"],
                        "training_time_seconds": training_time,
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                    })

                    current_process += 1
                    progress_bar.progress(min(current_process / total_process, 1.0))

    status_text.success("Eksperimen selesai.")

    dataset_summary_df = pd.DataFrame(dataset_summary)
    results_df = pd.DataFrame(results)
    selected_features_df = pd.DataFrame(selected_features_records)

    if results_df.empty:
        st.error("Tidak ada hasil eksperimen.")
        st.stop()

    # =========================================================
    # RINGKASAN HASIL
    # =========================================================

    best_by_dataset = (
        results_df
        .sort_values(
            by=["dataset", "f1_score", "auc", "mcc"],
            ascending=[True, False, False, False],
        )
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )

    best_overall = results_df.sort_values(
        by=["f1_score", "auc", "mcc"],
        ascending=[False, False, False],
    ).head(10)

    summary_by_model = (
        results_df
        .groupby("model")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "mcc",
                "training_time_seconds",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(by="f1_score", ascending=False)
    )

    summary_by_fs = (
        results_df
        .groupby("feature_selection")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "mcc",
                "training_time_seconds",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(by="f1_score", ascending=False)
    )

    fs_only_df = results_df[results_df["feature_selection"] != "No Feature Selection"]

    if fs_only_df.empty:
        summary_by_k = pd.DataFrame()
    else:
        summary_by_k = (
            fs_only_df
            .groupby("k_requested")[
                [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "auc",
                    "mcc",
                    "training_time_seconds",
                ]
            ]
            .mean()
            .reset_index()
            .sort_values(by="f1_score", ascending=False)
        )

    summary_by_combination = (
        results_df
        .groupby(["feature_selection", "k_requested", "model"])[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "mcc",
                "training_time_seconds",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(
            by=["f1_score", "auc", "mcc"],
            ascending=[False, False, False],
        )
    )

    # =========================================================
    # TAMPILAN HASIL
    # =========================================================

    st.success("✅ Hasil eksperimen berhasil dibuat.")

    st.subheader("📊 Ringkasan Dataset")
    st.dataframe(dataset_summary_df, use_container_width=True)

    best_row = best_overall.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model Terbaik", best_row["model"])
    col2.metric("F1-score", f"{best_row['f1_score']:.4f}")
    col3.metric("AUC", f"{best_row['auc']:.4f}")
    col4.metric("MCC", f"{best_row['mcc']:.4f}")

    st.markdown(
        f"""
        **Kombinasi terbaik** berdasarkan F1-score adalah:

        **{best_row['feature_selection']} | k = {best_row['k_requested']} | {best_row['model']}**

        Dataset: **{best_row['dataset']}**  
        F1-score: **{best_row['f1_score']:.4f}**  
        AUC: **{best_row['auc']:.4f}**  
        MCC: **{best_row['mcc']:.4f}**
        """
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "All Results",
        "Best Per Dataset",
        "Best Overall",
        "Summary Model",
        "Summary FS",
        "Summary K",
        "Summary Combination",
        "Selected Features",
    ])

    with tab1:
        st.dataframe(results_df, use_container_width=True)

    with tab2:
        st.dataframe(best_by_dataset, use_container_width=True)

    with tab3:
        st.dataframe(best_overall, use_container_width=True)

    with tab4:
        st.dataframe(summary_by_model, use_container_width=True)
        st.bar_chart(summary_by_model.set_index("model")[["f1_score", "auc", "mcc"]])

    with tab5:
        st.dataframe(summary_by_fs, use_container_width=True)
        st.bar_chart(summary_by_fs.set_index("feature_selection")[["f1_score", "auc", "mcc"]])

    with tab6:
        if summary_by_k.empty:
            st.info("Summary K tidak tersedia.")
        else:
            st.dataframe(summary_by_k, use_container_width=True)
            st.bar_chart(summary_by_k.set_index("k_requested")[["f1_score", "auc", "mcc"]])

    with tab7:
        st.dataframe(summary_by_combination, use_container_width=True)

    with tab8:
        st.dataframe(selected_features_df, use_container_width=True)

    # =========================================================
    # DOWNLOAD EXCEL
    # =========================================================

    excel_file = create_excel_file(
        dataset_summary_df=dataset_summary_df,
        results_df=results_df,
        best_by_dataset=best_by_dataset,
        best_overall=best_overall,
        summary_by_model=summary_by_model,
        summary_by_fs=summary_by_fs,
        summary_by_k=summary_by_k,
        summary_by_combination=summary_by_combination,
        selected_features_df=selected_features_df,
    )

    st.download_button(
        label="⬇️ Download Hasil Eksperimen Excel",
        data=excel_file,
        file_name="hasil_eksperimen_sdp.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )