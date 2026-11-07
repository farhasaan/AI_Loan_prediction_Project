from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score


# Lokasi model dan daftar fitur yang dipakai oleh aplikasi.
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "loan_approval_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL_FEATURES = ["education", "self_employed"]
ASSET_FEATURES = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
RAW_FEATURES = [*NUMERIC_FEATURES[:1], *CATEGORICAL_FEATURES[:1], CATEGORICAL_FEATURES[1], *NUMERIC_FEATURES[1:]]
LABELS = {0: "Ditolak", 1: "Disetujui"}

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Atur warna dan kontras komponen Streamlit.
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #f3f6fa; color: #172b3c; }
    [data-testid="stHeader"] { background: #fff; border-bottom: 1px solid #e2e8ef; }
    [data-testid="stToolbar"] { background: #fff; }
    [data-testid="stDeployButton"] button { color: #17324a !important; background: #fff !important; border: 1px solid #d4dee8 !important; border-radius: 8px !important; }
    [data-testid="stToolbar"] button { color: #40566b !important; }
    .block-container { max-width: 1360px; padding-top: 2.2rem; padding-bottom: 3rem; }
    [data-testid="stAppViewContainer"] > .main { background: #f3f6fa; }
    [data-testid="stAppViewContainer"] .main .block-container { color: #172b3c; }
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"] { color: #334b5f; }
    [data-testid="stAppViewContainer"] .main h1,
    [data-testid="stAppViewContainer"] .main h2,
    [data-testid="stAppViewContainer"] .main h3 { color: #142d42; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label { color: #263f54 !important; opacity: 1 !important; }
    [data-testid="stForm"] [data-testid="stMarkdownContainer"] { color: #52677a !important; }
    .hero { padding: 1.55rem 1.7rem; border-radius: 22px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1.4rem; }
    .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
    .hero p { color: #d9edf4; margin: 0; font-size: 1rem; }
    [data-testid="stTabs"] button { color: #40566b !important; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #c24132 !important; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea { background: #fff !important; color: #172b3c !important; border-color: #cbd5df !important; }
    [data-testid="stNumberInput"] button { color: #40566b !important; background: #f8fafc !important; }
    [data-testid="stForm"] { background: #fff; border: 1px solid #dce4ec; border-radius: 18px; padding: 1rem 1.1rem; }
    [data-testid="stFormSubmitButton"] button,
    .stDownloadButton button { background: #c24132 !important; border-color: #c24132 !important; color: #fff !important; }
    [data-testid="stFormSubmitButton"] button *,
    .stDownloadButton button * { color: #fff !important; }
    [data-testid="stAlert"] { color: #163b2b !important; }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * { color: #163b2b !important; opacity: 1 !important; }
    [data-testid="stAlert"] svg { fill: #207548 !important; }
    [data-testid="stProgress"] > div > div { background: #137c72 !important; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid #dce4ec; padding: .8rem 1rem; border-radius: 14px; }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #52667a !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #102a43 !important; }
    [data-testid="stSidebar"] { background: #202b36; }
    [data-testid="stSidebar"] :is(p, label, h1, h2, h3, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]) { color: #e6edf3 !important; }
    [data-testid="stSidebar"] div[data-testid="stMetric"] { background: #fff; }
    [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"] * { color: #52667a !important; opacity: 1 !important; }
    [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #102a43 !important; }
    .small-muted { color: #52677a; font-size: .88rem; }
    @media (max-width: 800px) { .block-container { padding: 1.2rem 1rem 2rem; } .hero { padding: 1.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    # Model dan metadata dimuat sekali agar tidak dibaca ulang setiap interaksi.
    missing = [path.name for path in [MODEL_PATH, METADATA_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"File model tidak ditemukan: {', '.join(missing)}")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, metadata, metrics


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # Samakan format nama kolom, misalnya "Loan Amount" menjadi "loan_amount".
    result = frame.copy()
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Nama kolom duplikat setelah normalisasi: {duplicates}")
    return result


def prepare_features(frame: pd.DataFrame, model_features: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    # Validasi input, bentuk fitur turunan, lalu susun sesuai kebutuhan model.
    data = normalize_columns(frame)
    missing = sorted(set(RAW_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    data = data[RAW_FEATURES].copy()
    invalid_counts: dict[str, int] = {}

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].map(
            lambda value: value.strip().lower() if isinstance(value, str) else value
        )
        allowed = {"graduate", "not graduate"} if column == "education" else {"yes", "no"}
        invalid = data[column].notna() & ~data[column].isin(allowed)
        invalid_counts[column] = int(invalid.sum())
        data.loc[invalid, column] = np.nan
        data[column] = data[column].replace({"": np.nan, "nan": np.nan, "null": np.nan, "none": np.nan})

    for column in NUMERIC_FEATURES:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce").replace([np.inf, -np.inf], np.nan)
        invalid = original.notna() & numeric.isna()
        invalid_counts[column] = int(invalid.sum())
        data[column] = numeric

    invalid_rules = {
        column: data[column].notna() & data[column].lt(0)
        for column in NUMERIC_FEATURES
    }
    for column in ["income_annum", "loan_amount", "loan_term"]:
        invalid_rules[column] = data[column].notna() & data[column].le(0)
    invalid_rules["cibil_score"] = data["cibil_score"].notna() & ~data["cibil_score"].between(300, 900)
    dependents = data["no_of_dependents"]
    invalid_rules["no_of_dependents"] = dependents.notna() & (
        dependents.lt(0) | dependents.mod(1).ne(0)
    )

    for column, invalid in invalid_rules.items():
        invalid_counts[column] = invalid_counts.get(column, 0) + int(invalid.sum())
        data.loc[invalid, column] = np.nan

    data["total_assets_value"] = data[ASSET_FEATURES].sum(axis=1, min_count=len(ASSET_FEATURES))
    data["loan_to_income_ratio"] = data["loan_amount"].div(data["income_annum"].replace(0, np.nan))
    data["assets_to_loan_ratio"] = data["total_assets_value"].div(data["loan_amount"].replace(0, np.nan))
    data["loan_amount_per_term"] = data["loan_amount"].div(data["loan_term"].replace(0, np.nan))
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.reindex(columns=model_features)
    return data, invalid_counts


def predict(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Kembalikan label prediksi dan peluang pengajuan disetujui.
    predictions = model.predict(features).astype(int)
    class_index = list(model.classes_).index(1)
    approval_probabilities = model.predict_proba(features)[:, class_index]
    return predictions, approval_probabilities


def format_amount(value: float) -> str:
    return f"{value:,.0f}"


try:
    # Hentikan aplikasi dengan pesan yang jelas jika aset model bermasalah.
    model, metadata, saved_metrics = load_assets()
except Exception as error:
    st.error(f"Model belum bisa dimuat: {error}")
    st.stop()

# Header aplikasi.
st.markdown(
    """
    <div class="hero">
      <h1>Loan Approval Prediction</h1>
      <p>Periksa estimasi status pengajuan pinjaman dari profil dan data finansial pemohon.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Panel samping menampilkan model dan ringkasan performanya.
with st.sidebar:
    st.subheader("Model")
    st.write(f"**Algoritma:** {metadata.get('selected_model', '—').replace('_', ' ').title()}")
    test_metrics = saved_metrics.get("metrics", {}).get("test", {})
    if test_metrics:
        st.metric("Akurasi test", f"{test_metrics.get('accuracy', 0):.1%}")
        st.metric("ROC AUC test", f"{test_metrics.get('roc_auc', 0):.3f}")
    st.caption("Skor berasal dari holdout dataset eksperimen.")
    st.divider()
    st.caption("Model memakai data historis Kaggle dan memberi estimasi untuk membantu peninjauan.")

# Pisahkan form prediksi, unggah batch, dan informasi model ke dalam tab.
tab_single, tab_batch, tab_about = st.tabs(["Satu Pengajuan", "Batch CSV", "Tentang Model"])

with tab_single:
    st.subheader("Data pemohon")
    st.caption("Masukkan nilai dengan satuan yang sama seperti dataset training.")

    with st.form("single_application_form"):
        applicant_col, loan_col, credit_col = st.columns(3)

        with applicant_col:
            dependents = st.number_input("Jumlah tanggungan", min_value=0, max_value=20, value=2, step=1)
            education = st.selectbox("Pendidikan", ["graduate", "not graduate"], format_func=lambda x: "Lulus kuliah" if x == "graduate" else "Belum lulus kuliah")
            self_employed = st.selectbox("Wiraswasta", ["no", "yes"], format_func=lambda x: "Tidak" if x == "no" else "Ya")
            income = st.number_input("Pendapatan tahunan", min_value=1.0, value=5_100_000.0, step=100_000.0, format="%.0f")

        with loan_col:
            loan_amount = st.number_input("Jumlah pinjaman", min_value=1.0, value=14_500_000.0, step=100_000.0, format="%.0f")
            loan_term = st.number_input("Jangka waktu (tahun)", min_value=1, max_value=50, value=10, step=1)
            cibil_score = st.number_input("Skor kredit (CIBIL)", min_value=300, max_value=900, value=650, step=1)

        with credit_col:
            residential_assets = st.number_input("Aset residensial", min_value=0.0, value=5_600_000.0, step=100_000.0, format="%.0f")
            commercial_assets = st.number_input("Aset komersial", min_value=0.0, value=3_700_000.0, step=100_000.0, format="%.0f")
            luxury_assets = st.number_input("Aset mewah", min_value=0.0, value=14_600_000.0, step=100_000.0, format="%.0f")
            bank_assets = st.number_input("Aset bank", min_value=0.0, value=4_600_000.0, step=100_000.0, format="%.0f")

        submitted = st.form_submit_button("Prediksi pengajuan", type="primary", use_container_width=True)

    if submitted:
        # Ubah nilai form menjadi satu baris data untuk diproses model.
        application = pd.DataFrame([{
            "no_of_dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets,
        }])
        features, _ = prepare_features(application, metadata["model_features"])
        prediction, probability = predict(model, features)
        status = LABELS[int(prediction[0])]
        st.divider()
        st.subheader("Hasil prediksi")
        result_col, probability_col = st.columns([1, 2])
        with result_col:
            if prediction[0] == 1:
                st.success(f"Estimasi: {status}")
            else:
                st.error(f"Estimasi: {status}")
        with probability_col:
            st.metric("Probabilitas disetujui", f"{probability[0]:.1%}")
            st.progress(float(probability[0]))
        st.caption("Hasil ini adalah prediksi model, bukan keputusan kredit final.")

with tab_batch:
    # Sediakan template, prediksi CSV, dan evaluasi jika label tersedia.
    st.subheader("Prediksi dari file CSV")
    st.caption("CSV harus memuat kolom fitur yang digunakan saat training. loan_id bersifat opsional.")
    template = pd.DataFrame(columns=RAW_FEATURES)
    st.download_button(
        "Unduh template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="loan_applications_template.csv",
        mime="text/csv",
    )
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        try:
            uploaded = pd.read_csv(uploaded_file)
            normalized_upload = normalize_columns(uploaded)
            features, invalid_counts = prepare_features(normalized_upload, metadata["model_features"])
            batch_predictions, batch_probabilities = predict(model, features)
            results = uploaded.copy()
            results["predicted_loan_status"] = pd.Series(batch_predictions, index=results.index).map(LABELS)
            results["approval_probability"] = batch_probabilities

            st.success(f"Prediksi selesai untuk {len(results):,} pengajuan.")
            flagged = {key: value for key, value in invalid_counts.items() if value}
            if flagged:
                st.warning("Sebagian nilai tidak valid diperlakukan sebagai missing dan diimputasi model.")
                st.json(flagged)
            st.dataframe(results.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Unduh hasil prediksi",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="loan_approval_predictions.csv",
                mime="text/csv",
            )

            if "loan_status" in normalized_upload.columns:
                actual = normalized_upload["loan_status"].astype("string").str.strip().str.lower()
                actual = actual.map({"approved": 1, "approve": 1, "1": 1, "yes": 1, "rejected": 0, "reject": 0, "0": 0, "no": 0})
                valid = actual.notna()
                if valid.any():
                    st.subheader("Evaluasi file berlabel")
                    m1, m2 = st.columns(2)
                    m1.metric("Akurasi", f"{accuracy_score(actual[valid], batch_predictions[valid.to_numpy()]):.1%}")
                    if actual[valid].nunique() == 2:
                        m2.metric("ROC AUC", f"{roc_auc_score(actual[valid], batch_probabilities[valid.to_numpy()]):.3f}")
        except Exception as error:
            st.error(f"File belum bisa diproses: {error}")

with tab_about:
    st.subheader("Ringkasan eksperimen")
    st.write({
        "Jumlah data training": metadata.get("train_rows"),
        "Jumlah data test": metadata.get("test_rows"),
        "Model": metadata.get("selected_model"),
        "Parameter terpilih": metadata.get("selected_parameters"),
    })
    st.write("Fitur yang digunakan")
    st.dataframe(pd.DataFrame({"feature": metadata.get("raw_features", [])}), use_container_width=True, hide_index=True)
    st.caption("Performa pada dataset eksperimen tidak menjamin performa yang sama pada data baru.")
