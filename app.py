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
    dan Kota Depok tahun **2018–2023** untuk membangun
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

    # Membersihkan nama kolom
    df.columns = (
        df.columns
        .str.strip()
    )

    # Mengubah kolom numerik
    kolom_numerik = [
        "kode_kabupaten_kota",
        "tahun",
        "jumlah_balita_stunting",
        "persentase_penduduk_miskin",
        "garis_kemiskinan",
        "persentase_sanitasi_layak",
        "jumlah_nakes_gizi"
    ]

    for kolom in kolom_numerik:

        if kolom in df.columns:

            df[kolom] = pd.to_numeric(
                df[kolom],
                errors="coerce"
            )

    # Menghapus data kosong
    df = df.dropna().copy()

    # Mengurutkan data
    df = df.sort_values(
        [
            "nama_kabupaten_kota",
            "tahun"
        ]
    ).reset_index(drop=True)

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
# LOAD EVALUASI MODEL
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
# LOAD SEMUA DATA DAN MODEL
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


target = "jumlah_balita_stunting"


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

    st.write(
        "Kolom yang tersedia:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


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
    "Periode Data: 2018–2023"
)

st.sidebar.write(
    "Periode Prediksi: 2025–2027"
)

st.sidebar.write(
    "Model: Random Forest Regressor"
)

st.sidebar.write(
    "Model Pembanding: Linear Regression"
)


# =========================================================
# DATASET AKTUAL
# =========================================================

st.subheader(
    "Data Aktual Kabupaten Indramayu "
    "dan Kota Depok Tahun 2018–2023"
)


st.dataframe(

    df[
        kolom_wajib
    ],

    use_container_width=True

)


# =========================================================
# PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df[
    fitur
]


df_actual = df.copy()


# Prediksi Random Forest
df_actual[
    "Prediksi Random Forest"
] = rf.predict(
    X_actual
)


# Prediksi Linear Regression
df_actual[
    "Prediksi Linear Regression"
] = lr.predict(
    X_actual
)


# Membulatkan hasil prediksi
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
    "Hasil Prediksi Model "
    "Tahun 2018–2023"
)


st.dataframe(

    df_actual[

        [

            "kode_kabupaten_kota",

            "nama_kabupaten_kota",

            "tahun",

            "jumlah_balita_stunting",

            "Prediksi Random Forest",

            "Prediksi Linear Regression"

        ]

    ],

    use_container_width=True

)


# =========================================================
# PREDIKSI TAHUN 2025–2027
# =========================================================

st.subheader(
    "Prediksi Jumlah Balita Stunting "
    "Tahun 2025–2027"
)


# Pastikan nama kolom hasil prediksi sesuai
df_future = df_future.copy()


# =========================================================
# VALIDASI KOLOM HASIL PREDIKSI
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
        "Kolom pada hasil_prediksi_2025_2027.csv "
        "tidak sesuai."
    )

    st.write(
        "Kolom yang hilang:"
    )

    st.write(
        kolom_prediksi_hilang
    )

    st.write(
        "Kolom yang tersedia:"
    )

    st.write(
        df_future.columns.tolist()
    )

    st.stop()


# =========================================================
# MEMASTIKAN NILAI PREDIKSI TIDAK NEGATIF
# =========================================================

df_future[
    "Prediksi Random Forest"
] = np.maximum(

    0,

    pd.to_numeric(

        df_future[
            "Prediksi Random Forest"
        ],

        errors="coerce"

    )

)


df_future[
    "Prediksi Linear Regression"
] = np.maximum(

    0,

    pd.to_numeric(

        df_future[
            "Prediksi Linear Regression"
        ],

        errors="coerce"

    )

)


# Membulatkan prediksi
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
# TABEL HASIL PREDIKSI
# =========================================================

st.dataframe(

    df_future[

        [

            "kode_kabupaten_kota",

            "nama_kabupaten_kota",

            "tahun",

            "Prediksi Random Forest",

            "Prediksi Linear Regression"

        ]

    ],

    use_container_width=True

)


# =========================================================
# VISUALISASI PREDIKSI RANDOM FOREST
# =========================================================

st.subheader(
    "Visualisasi Prediksi Random Forest "
    "Tahun 2025–2027"
)


# Membuat grafik berdasarkan wilayah
for wilayah in df_future[
    "nama_kabupaten_kota"
].unique():

    data_wilayah = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ]


    fig, ax = plt.subplots(

        figsize=(10, 5)

    )


    ax.bar(

        data_wilayah[
            "tahun"
        ].astype(str),

        data_wilayah[
            "Prediksi Random Forest"
        ]

    )


    ax.set_xlabel(
        "Tahun"
    )


    ax.set_ylabel(
        "Jumlah Balita Stunting"
    )


    ax.set_title(

        "Prediksi Random Forest - "

        + str(wilayah)

    )


    ax.grid(

        axis="y",

        alpha=0.3

    )


    st.pyplot(
        fig
    )


    plt.close(
        fig
    )


# =========================================================
# PERBANDINGAN MODEL
# =========================================================

st.subheader(
    "Perbandingan Prediksi "
    "Random Forest dan Linear Regression"
)


for wilayah in df_future[
    "nama_kabupaten_kota"
].unique():

    data_wilayah = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ]


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

        "Perbandingan Model - "

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
# TREN DATA AKTUAL DAN PREDIKSI
# =========================================================

st.subheader(
    "Grafik Tren Jumlah Balita Stunting "
    "Tahun 2018–2027"
)


for wilayah in df[
    "nama_kabupaten_kota"
].unique():

    data_aktual = df[

        df[
            "nama_kabupaten_kota"
        ] == wilayah

    ]


    data_prediksi = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ]


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

        label="Prediksi Random Forest"

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

        label="Prediksi Linear Regression"

    )


    ax.set_title(

        "Tren Jumlah Balita Stunting - "

        + str(wilayah)

        + " Tahun 2018–2027"

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

    "Evaluasi model Random Forest Regressor "
    "dan Linear Regression berdasarkan hasil "
    "pelatihan model."

)


# Menampilkan hasil evaluasi
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

    use_container_width=True

)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "Feature Importance Random Forest"
)


st.dataframe(

    df_feature,

    use_container_width=True

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
# KESIMPULAN PREDIKSI
# =========================================================

st.subheader(
    "Kesimpulan Prediksi"
)


# Mencari prediksi RF tertinggi
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


# Mencari prediksi LR tertinggi
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


st.info(

    f"**Random Forest Regressor:** "

    f"Prediksi tertinggi diperkirakan terjadi "

    f"di {wilayah_rf} pada tahun {tahun_rf} "

    f"dengan jumlah sekitar "

    f"{nilai_rf:,.0f} kasus. "

    f"**Linear Regression:** "

    f"Prediksi tertinggi diperkirakan terjadi "

    f"di {wilayah_lr} pada tahun {tahun_lr} "

    f"dengan jumlah sekitar "

    f"{nilai_lr:,.0f} kasus."

)
