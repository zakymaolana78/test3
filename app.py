import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Prediksi Jumlah Balita Stunting",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# JUDUL APLIKASI
# =========================================================

st.title(
    "Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu dan Kota Depok"
)

st.markdown(
    "Menggunakan algoritma "
    "**Random Forest Regressor** dan "
    "**Linear Regression**"
)

st.markdown(
    """
    Aplikasi ini menggunakan data Kabupaten Indramayu
    dan Kota Depok tahun **2018–2024** untuk membangun
    model prediksi dan melakukan estimasi jumlah balita
    stunting tahun **2025–2027**.
    """
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "dataset_indramayu_depok.csv"
    )

    # Membersihkan spasi pada nama kolom
    df.columns = (
        df.columns
        .str.strip()
    )

    # Kolom numerik
    kolom_numerik = [
        "kode_kabupaten_kota",
        "tahun",
        "jumlah_balita_stunting",
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "persentase_sanitasi_layak",
        "jumlah_nakes_gizi"
    ]

    # Mengubah kolom menjadi numerik
    for kolom in kolom_numerik:

        if kolom in df.columns:

            df[kolom] = pd.to_numeric(
                df[kolom],
                errors="coerce"
            )

    # Menghapus data kosong
    df = df.dropna().copy()

    # Mengurutkan berdasarkan wilayah dan tahun
    df = df.sort_values(
        [
            "nama_kabupaten_kota",
            "tahun"
        ]
    ).reset_index(
        drop=True
    )

    return df


# =========================================================
# LOAD MODEL RANDOM FOREST
# =========================================================

@st.cache_resource
def load_rf_model():

    return joblib.load(
        "random_forest_model.pkl"
    )


# =========================================================
# LOAD MODEL LINEAR REGRESSION
# =========================================================

@st.cache_resource
def load_lr_model():

    return joblib.load(
        "linear_regression_model.pkl"
    )


# =========================================================
# LOAD HASIL PREDIKSI 2025–2027
# =========================================================

@st.cache_data
def load_future_prediction():

    df_future = pd.read_csv(
        "hasil_prediksi_2025_2027.csv"
    )

    df_future.columns = (
        df_future.columns
        .str.strip()
    )

    return df_future


# =========================================================
# LOAD HASIL EVALUASI
# =========================================================

@st.cache_data
def load_evaluation():

    df_eval = pd.read_csv(
        "hasil_evaluasi_model.csv"
    )

    df_eval.columns = (
        df_eval.columns
        .str.strip()
    )

    return df_eval


# =========================================================
# LOAD FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def load_feature_importance():

    df_feature = pd.read_csv(
        "feature_importance.csv"
    )

    df_feature.columns = (
        df_feature.columns
        .str.strip()
    )

    return df_feature


# =========================================================
# LOAD SEMUA DATA
# =========================================================

try:

    df = load_data()

    rf = load_rf_model()

    lr = load_lr_model()

    df_future = load_future_prediction()

    df_eval = load_evaluation()

    df_feature = load_feature_importance()


except Exception as e:

    st.error(
        "Terjadi kesalahan saat memuat "
        "dataset, model, atau file hasil."
    )

    st.exception(e)

    st.stop()


# =========================================================
# FITUR DAN TARGET
# =========================================================

fitur = [

    "persentase_penduduk_miskin",

    "garis_kemiskinan",

    "persentase_sanitasi_layak",

    "jumlah_nakes_gizi"

]


target = (
    "jumlah_balita_stunting"
)


# =========================================================
# VALIDASI KOLOM DATASET
# =========================================================

kolom_wajib = [

    "kode_kabupaten_kota",

    "nama_kabupaten_kota",

    "tahun",

    "jumlah_balita_stunting",

    "persentase_penduduk_miskin",

    "garis_kemiskinan",

    "persentase_sanitasi_layak",

    "jumlah_nakes_gizi"

]


kolom_hilang = [

    kolom

    for kolom in kolom_wajib

    if kolom not in df.columns

]


if kolom_hilang:

    st.error(
        "Kolom berikut tidak ditemukan "
        "dalam dataset:"
    )

    st.write(
        kolom_hilang
    )

    st.stop()


# =========================================================
# VALIDASI RENTANG DATA
# =========================================================

tahun_min = int(
    df["tahun"].min()
)

tahun_max = int(
    df["tahun"].max()
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Informasi Penelitian"
)

st.sidebar.write(
    "Wilayah: Kabupaten Indramayu dan Kota Depok"
)

st.sidebar.write(
    f"Periode Data: {tahun_min}–{tahun_max}"
)

st.sidebar.write(
    "Periode Prediksi: 2025–2027"
)

st.sidebar.write(
    "Model Utama: Random Forest Regressor"
)

st.sidebar.write(
    "Model Pembanding: Linear Regression"
)


# =========================================================
# PERINGATAN JIKA DATA BELUM SAMPAI 2024
# =========================================================

if tahun_max < 2024:

    st.warning(
        f"Dataset yang sedang digunakan hanya memiliki "
        f"data sampai tahun {tahun_max}. "
        f"Pastikan dataset_indramayu_depok.csv "
        f"sudah memiliki data tahun 2024 untuk "
        f"Indramayu dan Kota Depok."
    )


# =========================================================
# DATA AKTUAL
# =========================================================

st.subheader(
    "Data Aktual 2018–2024"
)


st.dataframe(

    df[
        kolom_wajib
    ],

    use_container_width=True,

    hide_index=True

)


# =========================================================
# PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df[
    fitur
]


df_actual = df.copy()


# Random Forest
df_actual[
    "Prediksi Random Forest"
] = rf.predict(
    X_actual
)


# Linear Regression
df_actual[
    "Prediksi Linear Regression"
] = lr.predict(
    X_actual
)


# Membulatkan hasil
df_actual[
    "Prediksi Random Forest"
] = (

    df_actual[
        "Prediksi Random Forest"
    ]

    .round(0)

)


df_actual[
    "Prediksi Linear Regression"
] = (

    df_actual[
        "Prediksi Linear Regression"
    ]

    .round(0)

)


# =========================================================
# HASIL PREDIKSI DATA AKTUAL
# =========================================================

st.subheader(
    "Hasil Prediksi Data Aktual 2018–2024"
)


st.dataframe(

    df_actual[

        [

            "nama_kabupaten_kota",

            "tahun",

            "jumlah_balita_stunting",

            "Prediksi Random Forest",

            "Prediksi Linear Regression"

        ]

    ],

    use_container_width=True,

    hide_index=True

)


# =========================================================
# HASIL PREDIKSI 2025–2027
# =========================================================

st.subheader(
    "Prediksi Jumlah Balita Stunting 2025–2027"
)


df_future = df_future.copy()


# =========================================================
# VALIDASI HASIL PREDIKSI
# =========================================================

kolom_prediksi_wajib = [

    "kode_kabupaten_kota",

    "nama_kabupaten_kota",

    "tahun",

    "Prediksi Random Forest",

    "Prediksi Linear Regression"

]


kolom_prediksi_hilang = [

    kolom

    for kolom in kolom_prediksi_wajib

    if kolom not in df_future.columns

]


if kolom_prediksi_hilang:

    st.error(
        "Kolom hasil prediksi tidak sesuai."
    )

    st.write(
        "Kolom yang hilang:"
    )

    st.write(
        kolom_prediksi_hilang
    )

    st.stop()


# =========================================================
# KONVERSI NILAI PREDIKSI
# =========================================================

df_future[
    "Prediksi Random Forest"
] = pd.to_numeric(

    df_future[
        "Prediksi Random Forest"
    ],

    errors="coerce"

)


df_future[
    "Prediksi Linear Regression"
] = pd.to_numeric(

    df_future[
        "Prediksi Linear Regression"
    ],

    errors="coerce"

)


# =========================================================
# RANDOM FOREST TIDAK BOLEH NEGATIF
# =========================================================

df_future[
    "Prediksi Random Forest"
] = np.maximum(

    0,

    df_future[
        "Prediksi Random Forest"
    ]

)


# =========================================================
# MEMBULATKAN HASIL
# =========================================================

df_future[
    "Prediksi Random Forest"
] = (

    df_future[
        "Prediksi Random Forest"
    ]

    .round(0)

)


df_future[
    "Prediksi Linear Regression"
] = (

    df_future[
        "Prediksi Linear Regression"
    ]

    .round(0)

)


# =========================================================
# TABEL PREDIKSI MINIMALIS
# =========================================================

st.dataframe(

    df_future[

        [

            "nama_kabupaten_kota",

            "tahun",

            "Prediksi Random Forest",

            "Prediksi Linear Regression"

        ]

    ],

    use_container_width=True,

    hide_index=True

)


# =========================================================
# GRAFIK PREDIKSI 2025–2027
# =========================================================

st.subheader(
    "Grafik Prediksi 2025–2027"
)


for wilayah in sorted(

    df_future[
        "nama_kabupaten_kota"
    ].unique()

):

    data_wilayah = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ].sort_values(
        "tahun"
    )


    fig, ax = plt.subplots(

        figsize=(10, 5)

    )


    ax.plot(

        data_wilayah[
            "tahun"
        ],

        data_wilayah[
            "Prediksi Random Forest"
        ],

        marker="o",

        label="Random Forest"

    )


    ax.plot(

        data_wilayah[
            "tahun"
        ],

        data_wilayah[
            "Prediksi Linear Regression"
        ],

        marker="s",

        linestyle="--",

        label="Linear Regression"

    )


    ax.set_title(

        "Prediksi Jumlah Balita Stunting - "

        + str(wilayah)

    )


    ax.set_xlabel(
        "Tahun"
    )


    ax.set_ylabel(
        "Jumlah Balita Stunting"
    )


    ax.legend()


    ax.grid(

        True,

        alpha=0.3

    )


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


# =========================================================
# GRAFIK TREN AKTUAL DAN PREDIKSI
# =========================================================

st.subheader(
    "Grafik Tren Aktual dan Prediksi 2018–2027"
)


for wilayah in sorted(

    df[
        "nama_kabupaten_kota"
    ].unique()

):

    data_aktual = df[

        df[
            "nama_kabupaten_kota"
        ] == wilayah

    ].sort_values(
        "tahun"
    )


    data_prediksi = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ].sort_values(
        "tahun"
    )


    fig, ax = plt.subplots(

        figsize=(10, 5)

    )


    # Data aktual
    ax.plot(

        data_aktual[
            "tahun"
        ],

        data_aktual[
            target
        ],

        marker="o",

        label="Data Aktual"

    )


    # Prediksi Random Forest
    ax.plot(

        data_prediksi[
            "tahun"
        ],

        data_prediksi[
            "Prediksi Random Forest"
        ],

        marker="s",

        linestyle="--",

        label="Random Forest"

    )


    # Prediksi Linear Regression
    ax.plot(

        data_prediksi[
            "tahun"
        ],

        data_prediksi[
            "Prediksi Linear Regression"
        ],

        marker="^",

        linestyle=":",

        label="Linear Regression"

    )


    ax.set_title(

        "Tren Jumlah Balita Stunting - "

        + str(wilayah)

        + " (2018–2027)"

    )


    ax.set_xlabel(
        "Tahun"
    )


    ax.set_ylabel(
        "Jumlah Balita Stunting"
    )


    ax.legend()


    ax.grid(

        True,

        alpha=0.3

    )


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


# =========================================================
# EVALUASI MODEL
# =========================================================

st.subheader(
    "Evaluasi Model"
)


st.caption(

    "Perbandingan performa Random Forest Regressor "
    "dan Linear Regression berdasarkan data "
    "historis tahun 2018–2024."

)


st.dataframe(

    df_eval.style.format({

        "R2 Score":
        "{:.4f}",

        "MAE":
        "{:,.2f}",

        "MSE":
        "{:,.2f}",

        "RMSE":
        "{:,.2f}"

    }),

    use_container_width=True,

    hide_index=True

)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "Feature Importance Random Forest"
)


st.dataframe(

    df_feature,

    use_container_width=True,

    hide_index=True

)


# =========================================================
# GRAFIK FEATURE IMPORTANCE
# =========================================================

fig3, ax3 = plt.subplots(

    figsize=(10, 5)

)


ax3.barh(

    df_feature[
        "Fitur"
    ],

    df_feature[
        "Importance"
    ]

)


ax3.set_xlabel(
    "Nilai Importance"
)


ax3.set_ylabel(
    "Variabel"
)


ax3.set_title(
    "Feature Importance Random Forest"
)


ax3.invert_yaxis()


ax3.grid(

    axis="x",

    alpha=0.3

)


st.pyplot(
    fig3
)


plt.close(
    fig3
)


# =========================================================
# KESIMPULAN
# =========================================================

st.subheader(
    "Kesimpulan Prediksi"
)


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

idx_rf = df_future[
    "Prediksi Random Forest"
].idxmax()


wilayah_rf = df_future.loc[

    idx_rf,

    "nama_kabupaten_kota"

]


tahun_rf = int(

    df_future.loc[

        idx_rf,

        "tahun"

    ]

)


nilai_rf = df_future.loc[

    idx_rf,

    "Prediksi Random Forest"

]


# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------

idx_lr = df_future[
    "Prediksi Linear Regression"
].idxmax()


wilayah_lr = df_future.loc[

    idx_lr,

    "nama_kabupaten_kota"

]


tahun_lr = int(

    df_future.loc[

        idx_lr,

        "tahun"

    ]

)


nilai_lr = df_future.loc[

    idx_lr,

    "Prediksi Linear Regression"

]


# ---------------------------------------------------------
# Menentukan model terbaik berdasarkan R2
# ---------------------------------------------------------

model_terbaik = None


if "R2 Score" in df_eval.columns:

    idx_terbaik = df_eval[
        "R2 Score"
    ].idxmax()

    model_terbaik = df_eval.loc[
        idx_terbaik,
        "Model"
    ]


# ---------------------------------------------------------
# Kesimpulan
# ---------------------------------------------------------

kesimpulan = (

    f"Berdasarkan hasil prediksi tahun 2025–2027, "
    f"Random Forest menghasilkan prediksi tertinggi "
    f"di {wilayah_rf} pada tahun {tahun_rf} "
    f"dengan estimasi sekitar {nilai_rf:,.0f} kasus. "
    f"Linear Regression menghasilkan prediksi tertinggi "
    f"di {wilayah_lr} pada tahun {tahun_lr} "
    f"dengan estimasi sekitar {nilai_lr:,.0f} kasus."
)


if model_terbaik is not None:

    kesimpulan += (

        f" Berdasarkan nilai R², model dengan performa "
        f"terbaik adalah {model_terbaik}."

    )


st.info(
    kesimpulan
)
