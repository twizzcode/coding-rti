import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


st.set_page_config(
    page_title="Software Defect Prediction Dashboard",
    page_icon="🧪",
    layout="wide"
)


def normalize_column_name(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def detect_target_column(df: pd.DataFrame):
    """
    Mencoba menebak kolom target pada dataset defect prediction.
    Dataset NASA MDP biasanya punya label seperti: defects, defect, bug, label, class.
    """
    candidates = [
        "defects", "defect", "bug", "bugs", "label", "class",
        "target", "fault", "faults", "is_defective", "problem"
    ]

    normalized_map = {normalize_column_name(c): c for c in df.columns}

    for cand in candidates:
        if cand in normalized_map:
            return normalized_map[cand]

    last_col = df.columns[-1]
    unique_count = df[last_col].nunique(dropna=True)
    if unique_count <= 5:
        return last_col

    return df.columns[-1]


def convert_target_to_binary(y: pd.Series) -> pd.Series:
    """
    Mengubah label target menjadi 0/1.
    Mendukung label: TRUE/FALSE, yes/no, defect/non-defect, Y/N, 1/0.
    Jika target numerik berisi jumlah bug, nilai > 0 dianggap defect.
    """
    y_copy = y.copy()

    if y_copy.dtype == "bool":
        return y_copy.astype(int)

    if pd.api.types.is_numeric_dtype(y_copy):
        unique_values = sorted(pd.Series(y_copy.dropna().unique()).tolist())
        if len(unique_values) == 2:
            return y_copy.astype(int)
        return (y_copy > 0).astype(int)

    y_str = y_copy.astype(str).str.strip().str.lower()

    positive_labels = {
        "true", "yes", "y", "1", "defect", "defective", "bug", "faulty", "problem", "positive"
    }
    negative_labels = {
        "false", "no", "n", "0", "non-defect", "non_defect", "clean", "not_defect", "negative"
    }

    mapped = []
    for val in y_str:
        if val in positive_labels:
            mapped.append(1)
        elif val in negative_labels:
            mapped.append(0)
        else:
            mapped.append(np.nan)

    mapped_series = pd.Series(mapped, index=y.index)

    if mapped_series.isna().sum() == 0:
        return mapped_series.astype(int)

    encoder = LabelEncoder()
    return pd.Series(encoder.fit_transform(y_str), index=y.index).astype(int)


def clean_dataset(df: pd.DataFrame, target_col: str, missing_strategy: str):
    """
    Membersihkan dataset:
    - hapus duplikasi kolom
    - pisahkan X dan y
    - ambil fitur numerik saja
    - tangani missing value
    - ubah target menjadi binary
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    y = convert_target_to_binary(df[target_col])
    X = df.drop(columns=[target_col])

    # Dataset NASA MDP umumnya berisi software metrics numerik.
    X = X.select_dtypes(include=[np.number])

    data = pd.concat([X, y.rename("target")], axis=1)

    if missing_strategy == "Drop rows":
        data = data.dropna()
    elif missing_strategy == "Fill with median":
        feature_cols = [c for c in data.columns if c != "target"]
        data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True))
        data = data.dropna(subset=["target"])
    else:
        feature_cols = [c for c in data.columns if c != "target"]
        data[feature_cols] = data[feature_cols].fillna(0)
        data = data.dropna(subset=["target"])

    X_clean = data.drop(columns=["target"])
    y_clean = data["target"].astype(int)

    # Hilangkan kolom konstan karena tidak informatif.
    nunique = X_clean.nunique(dropna=True)
    X_clean = X_clean.loc[:, nunique > 1]

    return X_clean, y_clean


def get_selector(method: str, k: int):
    if method == "Chi-Square":
        return SelectKBest(score_func=chi2, k=k)
    if method == "Mutual Information":
        return SelectKBest(score_func=mutual_info_classif, k=k)
    if method == "ANOVA F-test":
        return SelectKBest(score_func=f_classif, k=k)
    return None


def get_model(model_name: str, random_state: int, use_class_weight: bool):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced" if use_class_weight else None,
            n_jobs=-1,
        )

    if model_name == "Support Vector Machine":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced" if use_class_weight else None,
        )

    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost belum terinstall. Jalankan: pip install xgboost")

        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError("Model tidak dikenali.")


def evaluate_model(y_true, y_pred, y_proba=None):
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            result["AUC"] = roc_auc_score(y_true, y_proba)
        else:
            result["AUC"] = np.nan
    except Exception:
        result["AUC"] = np.nan

    return result


def run_experiment(
    X,
    y,
    feature_selection_method,
    k_value,
    model_name,
    test_size,
    random_state,
    use_class_weight
):
    """
    Menjalankan satu skenario eksperimen:
    stratified split -> scaling -> feature selection -> training -> evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    selected_features = list(X_train_scaled.columns)
    feature_score_df = pd.DataFrame()

    if feature_selection_method != "No Feature Selection":
        k = min(int(k_value), X_train_scaled.shape[1])
        selector = get_selector(feature_selection_method, k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)

        support = selector.get_support()
        selected_features = list(X_train_scaled.columns[support])

        if hasattr(selector, "scores_"):
            feature_score_df = pd.DataFrame({
                "Feature": X_train_scaled.columns,
                "Score": selector.scores_
            }).sort_values("Score", ascending=False)

    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
        k = "All"

    model = get_model(model_name, random_state, use_class_weight)
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_selected)[:, 1]
        except Exception:
            y_proba = None

    metrics = evaluate_model(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    summary = {
        "Feature Selection": feature_selection_method,
        "k": k,
        "Model": model_name,
        **metrics,
        "Selected Features": ", ".join(selected_features)
    }

    return summary, cm, selected_features, feature_score_df


def build_sample_dataset(n_samples=600, random_state=42):
    """
    Dataset dummy untuk uji tampilan.
    Ini BUKAN NASA MDP asli, hanya contoh agar aplikasi bisa dicoba tanpa file eksternal.
    """
    rng = np.random.default_rng(random_state)

    loc = rng.normal(40, 15, n_samples).clip(1, None)
    v_g = rng.normal(8, 4, n_samples).clip(1, None)
    ev_g = v_g + rng.normal(0, 2, n_samples)
    iv_g = v_g + rng.normal(0, 2, n_samples)
    n = rng.normal(120, 50, n_samples).clip(1, None)
    v = n * rng.normal(4, 1, n_samples).clip(0.5, None)
    l = rng.random(n_samples)
    d = rng.normal(20, 8, n_samples).clip(1, None)
    i = rng.normal(30, 12, n_samples).clip(1, None)
    e = v * d
    b = v / 3000
    t = e / 18
    lOCode = loc * rng.uniform(0.3, 0.7, n_samples)
    lOComment = loc * rng.uniform(0.0, 0.15, n_samples)
    uniq_Op = rng.integers(4, 40, n_samples)
    uniq_Opnd = rng.integers(4, 80, n_samples)
    total_Op = rng.integers(20, 300, n_samples)
    total_Opnd = rng.integers(20, 300, n_samples)
    branch_count = rng.integers(1, 50, n_samples)

    risk_score = (
        0.025 * loc +
        0.10 * v_g +
        0.00002 * e +
        0.03 * branch_count +
        rng.normal(0, 1, n_samples)
    )
    threshold = np.quantile(risk_score, 0.78)
    defects = (risk_score > threshold).astype(int)

    return pd.DataFrame({
        "loc": loc.round(3),
        "v(g)": v_g.round(3),
        "ev(g)": ev_g.round(3),
        "iv(g)": iv_g.round(3),
        "n": n.round(3),
        "v": v.round(3),
        "l": l.round(3),
        "d": d.round(3),
        "i": i.round(3),
        "e": e.round(3),
        "b": b.round(3),
        "t": t.round(3),
        "lOCode": lOCode.round(3),
        "lOComment": lOComment.round(3),
        "uniq_Op": uniq_Op,
        "uniq_Opnd": uniq_Opnd,
        "total_Op": total_Op,
        "total_Opnd": total_Opnd,
        "branchCount": branch_count,
        "defects": defects
    })


def dataset_summary(X: pd.DataFrame, y: pd.Series):
    total = len(y)
    defect_count = int((y == 1).sum())
    non_defect_count = int((y == 0).sum())
    defect_ratio = defect_count / total if total else 0

    return {
        "Jumlah Data": total,
        "Jumlah Fitur Numerik": X.shape[1],
        "Non-Defect": non_defect_count,
        "Defect": defect_count,
        "Defect Ratio": defect_ratio
    }


st.sidebar.title("🧪 SDP Dashboard")
page = st.sidebar.radio(
    "Menu",
    [
        "1. Dataset",
        "2. Single Experiment",
        "3. Run All Scenarios",
        "4. About"
    ]
)

st.title("Software Defect Prediction Dashboard")
st.caption(
    "Implementasi Streamlit untuk eksperimen Feature Selection dan Machine Learning "
    "pada Software Defect Prediction berbasis NASA MDP."
)

with st.sidebar.expander("Dataset", expanded=True):
    data_source = st.radio(
        "Sumber dataset",
        ["Upload CSV", "Use sample dataset"]
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if data_source == "Use sample dataset":
    df = build_sample_dataset()
else:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

if df is None:
    st.info("Upload dataset CSV terlebih dahulu atau gunakan sample dataset dari sidebar.")
    st.stop()

target_guess = detect_target_column(df)

with st.sidebar.expander("Preprocessing", expanded=True):
    target_col = st.selectbox(
        "Kolom target/label defect",
        options=list(df.columns),
        index=list(df.columns).index(target_guess) if target_guess in df.columns else len(df.columns) - 1
    )

    missing_strategy = st.selectbox(
        "Missing value",
        ["Drop rows", "Fill with median", "Fill with zero"]
    )

try:
    X, y = clean_dataset(df, target_col, missing_strategy)
except Exception as e:
    st.error(f"Gagal memproses dataset: {e}")
    st.stop()

if X.shape[1] == 0:
    st.error("Tidak ada fitur numerik yang bisa digunakan. Pastikan dataset berisi software metrics numerik.")
    st.stop()

if y.nunique() != 2:
    st.error("Target harus berupa klasifikasi biner: defect dan non-defect.")
    st.stop()

if min(y.value_counts()) < 2:
    st.error("Setiap kelas minimal harus memiliki 2 data agar stratified split bisa dilakukan.")
    st.stop()


if page == "1. Dataset":
    st.header("1. Dataset Overview")

    summary = dataset_summary(X, y)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jumlah Data", summary["Jumlah Data"])
    col2.metric("Jumlah Fitur", summary["Jumlah Fitur Numerik"])
    col3.metric("Non-Defect", summary["Non-Defect"])
    col4.metric("Defect", summary["Defect"])
    col5.metric("Defect Ratio", f"{summary['Defect Ratio']:.2%}")

    st.subheader("Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Fitur Numerik yang Digunakan")
    st.dataframe(pd.DataFrame({"Feature": X.columns}), use_container_width=True)

    st.subheader("Distribusi Kelas")
    class_dist = pd.DataFrame({
        "Class": ["Non-Defect", "Defect"],
        "Count": [summary["Non-Defect"], summary["Defect"]]
    })
    st.bar_chart(class_dist.set_index("Class"))

    st.subheader("Missing Value per Kolom")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df, use_container_width=True)


def experiment_sidebar():
    with st.sidebar.expander("Experiment Settings", expanded=True):
        fs_method = st.selectbox(
            "Feature Selection",
            ["No Feature Selection", "Chi-Square", "Mutual Information", "ANOVA F-test"]
        )

        k_value = st.selectbox("Jumlah fitur k", [5, 10, 15, 20])

        model_options = ["Random Forest", "Support Vector Machine"]
        if XGBOOST_AVAILABLE:
            model_options.append("XGBoost")
        else:
            st.warning("XGBoost belum tersedia. Install dengan: pip install xgboost")

        model_name = st.selectbox("Model", model_options)

        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", min_value=0, value=42)

        use_class_weight = st.checkbox(
            "Gunakan class_weight balanced",
            value=False,
            help="Aktifkan jika dataset sangat imbalanced. Untuk mendekati eksperimen dasar paper, biarkan OFF."
        )

    return fs_method, k_value, model_name, test_size, random_state, use_class_weight


if page == "2. Single Experiment":
    st.header("2. Single Experiment")

    fs_method, k_value, model_name, test_size, random_state, use_class_weight = experiment_sidebar()

    st.write(
        "Jalankan satu kombinasi eksperimen: metode feature selection, nilai k, dan algoritma machine learning."
    )

    if st.button("🚀 Jalankan Eksperimen", type="primary"):
        try:
            result, cm, selected_features, feature_score_df = run_experiment(
                X=X,
                y=y,
                feature_selection_method=fs_method,
                k_value=k_value,
                model_name=model_name,
                test_size=test_size,
                random_state=int(random_state),
                use_class_weight=use_class_weight
            )

            st.subheader("Hasil Evaluasi")

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{result['Accuracy']:.4f}")
            c2.metric("Precision", f"{result['Precision']:.4f}")
            c3.metric("Recall", f"{result['Recall']:.4f}")

            c4, c5, c6 = st.columns(3)
            c4.metric("F1-score", f"{result['F1-score']:.4f}")
            c5.metric("AUC", f"{result['AUC']:.4f}" if not pd.isna(result["AUC"]) else "N/A")
            c6.metric("MCC", f"{result['MCC']:.4f}")

            st.subheader("Ringkasan Skenario")
            result_df = pd.DataFrame([result])
            st.dataframe(result_df, use_container_width=True)

            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Non-Defect", "Actual Defect"],
                columns=["Predicted Non-Defect", "Predicted Defect"]
            )
            st.dataframe(cm_df, use_container_width=True)

            st.subheader("Fitur yang Digunakan Model")
            st.write(selected_features)

            if not feature_score_df.empty:
                st.subheader("Ranking Feature Score")
                st.dataframe(feature_score_df, use_container_width=True)
                st.bar_chart(feature_score_df.head(15).set_index("Feature")["Score"])

        except Exception as e:
            st.error(f"Terjadi error saat eksperimen: {e}")


if page == "3. Run All Scenarios":
    st.header("3. Run All Scenarios")

    st.write(
        "Menu ini menjalankan seluruh skenario seperti rancangan paper: "
        "3 feature selection × 4 nilai k × 3 model + baseline no feature selection."
    )

    with st.sidebar.expander("Run All Settings", expanded=True):
        test_size_all = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="test_size_all")
        random_state_all = st.number_input("Random state", min_value=0, value=42, key="random_state_all")
        use_class_weight_all = st.checkbox(
            "Gunakan class_weight balanced",
            value=False,
            key="cw_all"
        )

    model_list = ["Random Forest", "Support Vector Machine"]
    if XGBOOST_AVAILABLE:
        model_list.append("XGBoost")

    if st.button("🚀 Jalankan Semua Skenario", type="primary"):
        scenarios = []

        feature_selection_methods = ["ANOVA F-test", "Chi-Square", "Mutual Information"]
        k_values = [5, 10, 15, 20]

        progress_bar = st.progress(0)
        total_runs = (len(feature_selection_methods) * len(k_values) * len(model_list)) + len(model_list)
        run_count = 0

        for fs in feature_selection_methods:
            for k in k_values:
                for model in model_list:
                    try:
                        result, _, _, _ = run_experiment(
                            X=X,
                            y=y,
                            feature_selection_method=fs,
                            k_value=k,
                            model_name=model,
                            test_size=test_size_all,
                            random_state=int(random_state_all),
                            use_class_weight=use_class_weight_all
                        )
                        scenarios.append(result)
                    except Exception as e:
                        scenarios.append({
                            "Feature Selection": fs,
                            "k": k,
                            "Model": model,
                            "Error": str(e)
                        })

                    run_count += 1
                    progress_bar.progress(run_count / total_runs)

        for model in model_list:
            try:
                result, _, _, _ = run_experiment(
                    X=X,
                    y=y,
                    feature_selection_method="No Feature Selection",
                    k_value=0,
                    model_name=model,
                    test_size=test_size_all,
                    random_state=int(random_state_all),
                    use_class_weight=use_class_weight_all
                )
                scenarios.append(result)
            except Exception as e:
                scenarios.append({
                    "Feature Selection": "No Feature Selection",
                    "k": "All",
                    "Model": model,
                    "Error": str(e)
                })

            run_count += 1
            progress_bar.progress(run_count / total_runs)

        results_df = pd.DataFrame(scenarios)

        metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "AUC", "MCC"]
        for col in metric_cols:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

        st.subheader("Tabel Semua Skenario")
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Top 10 Berdasarkan F1-score")
        if "F1-score" in results_df.columns:
            top_f1 = results_df.sort_values(["F1-score", "MCC"], ascending=False).head(10)
            st.dataframe(top_f1, use_container_width=True)

        st.subheader("Rata-rata Berdasarkan Model")
        available_metrics = [c for c in metric_cols if c in results_df.columns]
        if available_metrics:
            avg_model = results_df.groupby("Model")[available_metrics].mean().reset_index()
            st.dataframe(avg_model, use_container_width=True)

            st.subheader("Rata-rata Berdasarkan Feature Selection")
            avg_fs = results_df.groupby("Feature Selection")[available_metrics].mean().reset_index()
            st.dataframe(avg_fs, use_container_width=True)

            if "F1-score" in avg_model.columns:
                st.subheader("Perbandingan F1-score per Model")
                chart_df = avg_model[["Model", "F1-score"]].set_index("Model")
                st.bar_chart(chart_df)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download hasil eksperimen CSV",
            data=csv,
            file_name="sdp_experiment_results.csv",
            mime="text/csv"
        )


if page == "4. About":
    st.header("4. About This App")

    st.markdown(
        """
        Aplikasi ini dibuat sebagai implementasi web dari paper:

        **Comparative Analysis of Feature Selection Methods and Machine Learning Algorithms for Software Defect Prediction Using NASA MDP Datasets**

        ### Tujuan Aplikasi
        Aplikasi ini membantu menjalankan ulang eksperimen software defect prediction secara interaktif.
        Pengguna dapat mengunggah dataset NASA MDP dalam format CSV, memilih metode feature selection,
        memilih jumlah fitur, memilih algoritma machine learning, lalu melihat hasil evaluasi model.

        ### Metode Feature Selection
        - No Feature Selection
        - Chi-Square
        - Mutual Information
        - ANOVA F-test

        ### Algoritma Machine Learning
        - Random Forest
        - Support Vector Machine
        - XGBoost

        ### Metrik Evaluasi
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - AUC
        - MCC

        ### Catatan Penting
        Dataset NASA MDP memiliki karakteristik class imbalance. Karena itu, evaluasi tidak cukup hanya
        menggunakan accuracy. Metrik seperti recall, F1-score, AUC, dan MCC lebih penting untuk menilai
        kemampuan model dalam mendeteksi modul defect.
        """
    )

    st.info(
        "Untuk hasil yang mendekati paper, gunakan dataset NASA MDP asli seperti JM1, KC1, PC1, CM1, dan KC3."
    )
