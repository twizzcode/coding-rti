import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


from google.colab import drive
drive.mount('/content/drive')


base_path = "/content/drive/MyDrive/MDP/D''"
output_path = "/content/drive/MyDrive/MDP"

os.makedirs(output_path, exist_ok=True)

print("Isi folder dataset:")
print(os.listdir(base_path))


dataset_files = {
    "JM1": f"{base_path}/JM1.arff",
    "KC1": f"{base_path}/KC1.arff",
    "PC1": f"{base_path}/PC1.arff",
    "CM1": f"{base_path}/CM1.arff",
    "KC3": f"{base_path}/KC3.arff"
}


for name, path in dataset_files.items():
    print(name, "=>", os.path.exists(path), path)


from scipy.io import arff

def read_dataset(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return pd.read_excel(file_path)

    elif file_path.endswith(".arff"):
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        # ubah kolom bytes menjadi string
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )

        return df

    else:
        raise ValueError(f"Format file tidak didukung: {file_path}")


df = read_dataset("/content/drive/MyDrive/MDP/D''/JM1.arff")

print(df.shape)
print(df.head())
print(df.columns)
print(df.dtypes)


def detect_target_column(df):
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
        "Label"
    ]

    for col in possible_targets:
        if col in df.columns:
            return col

    # fallback: jika tidak ketemu, pakai kolom terakhir sebagai target
    return df.columns[-1]



def preprocess_dataset(df):
    df = df.copy()

    # rapikan nama kolom
    df.columns = df.columns.astype(str).str.strip()

    target_col = detect_target_column(df)

    # hapus kolom ID/name/module kalau ada
    possible_id_cols = ["id", "ID", "name", "Name", "module", "Module"]
    for col in possible_id_cols:
        if col in df.columns and col != target_col:
            df = df.drop(columns=[col])

    # target
    y = df[target_col]

    # handle bytes
    y = y.apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # ubah target ke 0/1
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
            "buggy": 1,
            "faulty": 1,
            "non-defect": 0,
            "non_defect": 0,
            "clean": 0,
            "not_buggy": 0,
            "not faulty": 0,
            "0": 0,
            "1": 1
        }

        if set(y.unique()).issubset(set(mapping.keys())):
            y = y.map(mapping)
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        y = y.astype(int)

    # fitur
    X = df.drop(columns=[target_col])

    # handle bytes di fitur
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    # ubah fitur ke numerik
    X = X.apply(pd.to_numeric, errors="coerce")

    # hapus kolom kosong semua
    X = X.dropna(axis=1, how="all")

    # handle inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # isi missing value dengan median
    X = X.fillna(X.median())

    # pastikan target valid
    valid_idx = pd.Series(y).notna()
    X = X.loc[valid_idx]
    y = pd.Series(y).loc[valid_idx].astype(int).values

    return X, y, target_col


def apply_feature_selection(method, X_train, y_train, X_test, k=None):
    if method == "none":
        selected_features = list(X_train.columns)
        return X_train.values, X_test.values, selected_features

    if k is None:
        raise ValueError("Nilai k harus diisi untuk metode feature selection selain none.")

    k = min(k, X_train.shape[1])

    if method == "chi2":
        # Chi-Square butuh nilai non-negatif
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
        raise ValueError(f"Metode feature selection tidak dikenal: {method}")

    selected_features = list(X_train.columns[selector.get_support()])
    return X_train_selected, X_test_selected, selected_features



def get_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss"
        )
    }

    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = np.nan

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc": auc,
        "mcc": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred


dataset_summary = []

for dataset_name, file_path in dataset_files.items():
    if not os.path.exists(file_path):
        print(f"File tidak ditemukan: {file_path}")
        continue

    df = read_dataset(file_path)
    X, y, target_col = preprocess_dataset(df)

    class_dist = pd.Series(y).value_counts().to_dict()

    dataset_summary.append({
        "dataset": dataset_name,
        "target_column": target_col,
        "num_data": X.shape[0],
        "num_features": X.shape[1],
        "class_0": class_dist.get(0, 0),
        "class_1": class_dist.get(1, 0),
        "defect_ratio": class_dist.get(1, 0) / X.shape[0]
    })

dataset_summary_df = pd.DataFrame(dataset_summary)
dataset_summary_df


feature_methods = {
    "No Feature Selection": "none",
    "Chi-Square": "chi2",
    "Mutual Information": "mutual_info",
    "ANOVA F-test": "anova"
}

k_values = [5, 10, 15, 20]

results = []
selected_features_records = []

for dataset_name, file_path in dataset_files.items():

    if not os.path.exists(file_path):
        print(f"\nFile tidak ditemukan: {file_path}")
        continue

    print("=" * 100)
    print(f"Memproses dataset: {dataset_name}")

    df = read_dataset(file_path)
    X, y, target_col = preprocess_dataset(df)

    print(f"Target column        : {target_col}")
    print(f"Jumlah data          : {X.shape[0]}")
    print(f"Jumlah fitur awal    : {X.shape[1]}")
    print(f"Distribusi kelas     : {pd.Series(y).value_counts().to_dict()}")

    if len(np.unique(y)) < 2:
        print(f"Dataset {dataset_name} dilewati karena hanya memiliki 1 kelas.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    for fs_name, fs_method in feature_methods.items():

        # No Feature Selection tidak perlu variasi k
        if fs_method == "none":
            current_k_values = ["all"]
        else:
            current_k_values = k_values

        for k in current_k_values:

            print(f"\nFeature Selection: {fs_name} | k = {k}")

            if fs_method == "none":
                X_train_fs, X_test_fs, selected_features = apply_feature_selection(
                    fs_method,
                    X_train,
                    y_train,
                    X_test,
                    k=None
                )
                actual_k = X_train.shape[1]
            else:
                X_train_fs, X_test_fs, selected_features = apply_feature_selection(
                    fs_method,
                    X_train,
                    y_train,
                    X_test,
                    k=k
                )
                actual_k = len(selected_features)

            selected_features_records.append({
                "dataset": dataset_name,
                "feature_selection": fs_name,
                "k_requested": k,
                "k_actual": actual_k,
                "selected_features": ", ".join(selected_features)
            })

            models = get_models()

            for model_name, model in models.items():

                X_train_model = X_train_fs
                X_test_model = X_test_fs

                # scaling khusus SVM
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
                    "tp": tp
                })

                print(
                    f"{model_name} | "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"Prec: {metrics['precision']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | "
                    f"F1: {metrics['f1_score']:.4f} | "
                    f"AUC: {metrics['auc']:.4f} | "
                    f"MCC: {metrics['mcc']:.4f}"
                )

results_df = pd.DataFrame(results)
features_df = pd.DataFrame(selected_features_records)

results_df


best_by_dataset = (
    results_df
    .sort_values(
        by=["dataset", "f1_score", "auc", "mcc"],
        ascending=[True, False, False, False]
    )
    .groupby("dataset")
    .head(1)
    .reset_index(drop=True)
)

best_by_dataset


best_overall = results_df.sort_values(
    by=["f1_score", "auc", "mcc"],
    ascending=[False, False, False]
).head(10)

best_overall



summary_by_model = results_df.groupby("model")[
    ["accuracy", "precision", "recall", "f1_score", "auc", "mcc", "training_time_seconds"]
].mean().reset_index()

summary_by_model.sort_values(by="f1_score", ascending=False)


fs_only_df = results_df[results_df["feature_selection"] != "No Feature Selection"].copy()

summary_by_k = fs_only_df.groupby("k_requested")[
    ["accuracy", "precision", "recall", "f1_score", "auc", "mcc", "training_time_seconds"]
].mean().reset_index()

summary_by_k.sort_values(by="f1_score", ascending=False)


summary_by_combination = results_df.groupby(["feature_selection", "k_requested", "model"])[
    ["accuracy", "precision", "recall", "f1_score", "auc", "mcc", "training_time_seconds"]
].mean().reset_index()

summary_by_combination.sort_values(by="f1_score", ascending=False)


output_file = f"{output_path}/hasil_eksperimen_k_5_10_15_20.xlsx"

with pd.ExcelWriter(output_file) as writer:
    dataset_summary_df.to_excel(writer, sheet_name="Dataset Summary", index=False)
    results_df.to_excel(writer, sheet_name="All Results", index=False)
    best_by_dataset.to_excel(writer, sheet_name="Best Per Dataset", index=False)
    best_overall.to_excel(writer, sheet_name="Best Overall", index=False)
    summary_by_model.to_excel(writer, sheet_name="Summary Model", index=False)
    summary_by_fs.to_excel(writer, sheet_name="Summary FS", index=False)
    summary_by_k.to_excel(writer, sheet_name="Summary K", index=False)
    summary_by_combination.to_excel(writer, sheet_name="Summary Combination", index=False)
    features_df.to_excel(writer, sheet_name="Selected Features", index=False)

print("File berhasil disimpan di:")
print(output_file)



print("=== MODEL TERBAIK BERDASARKAN RATA-RATA F1-SCORE ===")
best_model = summary_by_model.sort_values(by="f1_score", ascending=False).iloc[0]
print(
    f"{best_model['model']} dengan rata-rata F1-score = {best_model['f1_score']:.4f}, "
    f"AUC = {best_model['auc']:.4f}, MCC = {best_model['mcc']:.4f}"
)

print("\n=== FEATURE SELECTION TERBAIK BERDASARKAN RATA-RATA F1-SCORE ===")
best_fs = summary_by_fs.sort_values(by="f1_score", ascending=False).iloc[0]
print(
    f"{best_fs['feature_selection']} dengan rata-rata F1-score = {best_fs['f1_score']:.4f}, "
    f"AUC = {best_fs['auc']:.4f}, MCC = {best_fs['mcc']:.4f}"
)

print("\n=== NILAI K TERBAIK BERDASARKAN RATA-RATA F1-SCORE ===")
best_k = summary_by_k.sort_values(by="f1_score", ascending=False).iloc[0]
print(
    f"k = {best_k['k_requested']} dengan rata-rata F1-score = {best_k['f1_score']:.4f}, "
    f"AUC = {best_k['auc']:.4f}, MCC = {best_k['mcc']:.4f}"
)

print("\n=== KOMBINASI TERBAIK SECARA RATA-RATA ===")
best_comb = summary_by_combination.sort_values(
    by=["f1_score", "auc", "mcc"],
    ascending=[False, False, False]
).iloc[0]

print(
    f"{best_comb['feature_selection']} | k = {best_comb['k_requested']} | {best_comb['model']} "
    f"dengan rata-rata F1-score = {best_comb['f1_score']:.4f}, "
    f"AUC = {best_comb['auc']:.4f}, MCC = {best_comb['mcc']:.4f}"
)