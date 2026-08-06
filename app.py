import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Prediksi Jumlah Balita Stunting",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 2. JUDUL APLIKASI
# =========================================================

st.title(
    "Prediksi Jumlah Balita Stunting "
    "Kabupaten Indramayu dan Kota Depok"
)

st.markdown(
    "Menggunakan algoritma **Random Forest Regressor** "
    "dan **Linear Regression**"
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
# 3. LOAD DATASET
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

    # Mengubah koma desimal menjadi titik
    # CATATAN PERBAIKAN:
    # Sebelumnya kondisi ini hanya mengecek dtype == "object".
    # Di pandas versi baru (pandas 3.x), kolom teks bisa
    # bertipe StringDtype (bukan object), sehingga kondisi lama
    # tidak pernah terpenuhi dan koma desimal tidak pernah
    # diganti menjadi titik. Akibatnya nilai seperti "97,96"
    # gagal dikonversi ke angka, menjadi NaN, lalu terhapus oleh
    # dropna() -- inilah yang menyebabkan data tahun 2024 hilang.
    # Fix: gunakan pd.api.types.is_string_dtype() agar mengenali
    # baik dtype "object" (pandas lama) maupun "string" (pandas baru).
    for kolom in df.columns:

        if df[kolom].dtype == "object" or pd.api.types.is_string_dtype(df[kolom]):

            df[kolom] = (
                df[kolom]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
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


    # Konversi kolom numerik
    for kolom in kolom_numerik:

        if kolom in df.columns:

            df[kolom] = pd.to_numeric(

                df[kolom],

                errors="coerce"

            )


    # Menghapus data kosong
    df = df.dropna().copy()


    # Memastikan tahun berupa integer
    df["tahun"] = (
        df["tahun"]
        .astype(int)
    )


    # Memastikan kode wilayah berupa integer
    df["kode_kabupaten_kota"] = (
        df["kode_kabupaten_kota"]
        .astype(int)
    )


    # Mengurutkan data
    df = (

        df

        .sort_values(

            [
                "nama_kabupaten_kota",

                "tahun"

            ]

        )

        .reset_index(drop=True)

    )


    return df


# =========================================================
# 4. LOAD MODEL RANDOM FOREST
# =========================================================

@st.cache_resource
def load_rf_model():

    return joblib.load(
        "random_forest_model.pkl"
    )


# =========================================================
# 5. LOAD MODEL LINEAR REGRESSION
# =========================================================

@st.cache_resource
def load_lr_model():

    return joblib.load(
        "linear_regression_model.pkl"
    )


# =========================================================
# 6. LOAD HASIL PREDIKSI 2025–2027
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
# 7. LOAD EVALUASI MODEL
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
# 8. LOAD FEATURE IMPORTANCE
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
# 9. LOAD SEMUA DATA DAN MODEL
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
# 10. FITUR DAN TARGET
# =========================================================

fitur = [

    "persentase_penduduk_miskin",

    "garis_kemiskinan",

    "persentase_sanitasi_layak",

    "jumlah_nakes_gizi"

]


target = "jumlah_balita_stunting"


# =========================================================
# 11. VALIDASI KOLOM DATASET
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
# 12. SIDEBAR
# =========================================================

st.sidebar.header(
    "Informasi Penelitian"
)

st.sidebar.write(
    "Wilayah: Kabupaten Indramayu dan Kota Depok"
)

st.sidebar.write(
    "Periode Data: 2018–2024"
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


# Informasi jumlah data
st.sidebar.success(
    f"Total Data: {len(df)} baris"
)

st.sidebar.success(
    f"Rentang Tahun: "
    f"{df['tahun'].min()}–{df['tahun'].max()}"
)


# =========================================================
# 13. DATA AKTUAL 2018–2024
# =========================================================

df_aktual = df[

    (df["tahun"] >= 2018)

    &

    (df["tahun"] <= 2024)

].copy()


st.subheader(
    "Data Aktual Kabupaten Indramayu "
    "dan Kota Depok Tahun 2018–2024"
)


st.write(
    f"Jumlah data aktual: **{len(df_aktual)} baris**"
)


st.dataframe(

    df_aktual[

        kolom_wajib

    ],

    use_container_width=True

)


# =========================================================
# 14. PREDIKSI DATA AKTUAL
# =========================================================

X_actual = df_aktual[

    fitur

]


df_actual = df_aktual.copy()


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
# 15. HASIL PREDIKSI DATA AKTUAL
# =========================================================

st.subheader(
    "Hasil Prediksi Data Aktual Tahun 2018–2024"
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
# 16. PREDIKSI TAHUN 2025–2027
# =========================================================

st.subheader(
    "Prediksi Jumlah Balita Stunting Tahun 2025–2027"
)


df_future = df_future.copy()


# Memastikan tahun berupa integer
df_future["tahun"] = pd.to_numeric(

    df_future["tahun"],

    errors="coerce"

).astype(int)


# Memastikan nilai prediksi numerik
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


# Membulatkan hasil
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
# 17. TABEL PREDIKSI MINIMALIS
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

    use_container_width=True

)


# =========================================================
# 18. GRAFIK PREDIKSI 2025–2027
# =========================================================

st.subheader(
    "Grafik Prediksi Tahun 2025–2027"
)


# Batas skala sumbu Y disamakan untuk semua wilayah
# agar grafik Kabupaten Indramayu dan Kota Depok
# dapat dibandingkan pada rentang yang sama
# (minimal sampai 10000, mengikuti skala Kota Depok).

y_min_prediksi = min(

    df_future["Prediksi Random Forest"].min(),

    df_future["Prediksi Linear Regression"].min()

)


y_max_prediksi = max(

    df_future["Prediksi Random Forest"].max(),

    df_future["Prediksi Linear Regression"].max(),

    10000

)


# Menambahkan padding di atas dan di bawah
padding_prediksi = (y_max_prediksi - y_min_prediksi) * 0.05

batas_bawah_prediksi = y_min_prediksi - padding_prediksi

batas_atas_prediksi = y_max_prediksi + padding_prediksi


for wilayah in sorted(

    df_future[
        "nama_kabupaten_kota"
    ].unique()

):

    data_wilayah = df_future[

        df_future[
            "nama_kabupaten_kota"
        ] == wilayah

    ]


    fig, ax = plt.subplots(

        figsize=(10, 5)

    )


    ax.plot(

        data_wilayah["tahun"],

        data_wilayah[
            "Prediksi Random Forest"
        ],

        marker="o",

        label="Random Forest"

    )


    ax.plot(

        data_wilayah["tahun"],

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


    # Skala sumbu Y disamakan (lihat perhitungan di atas)
    ax.set_ylim(

        batas_bawah_prediksi,

        batas_atas_prediksi

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
# 19. GRAFIK TREN DATA AKTUAL DAN PREDIKSI
# =========================================================

st.subheader(
    "Grafik Tren Jumlah Balita Stunting "
    "Tahun 2018–2027"
)


for wilayah in sorted(

    df_aktual[
        "nama_kabupaten_kota"
    ].unique()

):

    data_aktual = df_aktual[

        df_aktual[
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

        data_aktual["tahun"],

        data_aktual[target],

        marker="o",

        label="Data Aktual"

    )


    # Prediksi Random Forest
    ax.plot(

        data_prediksi["tahun"],

        data_prediksi[
            "Prediksi Random Forest"
        ],

        marker="s",

        linestyle="--",

        label="Prediksi Random Forest"

    )


    # Prediksi Linear Regression
    ax.plot(

        data_prediksi["tahun"],

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
# 20. EVALUASI MODEL
# =========================================================

st.subheader(
    "Evaluasi Model"
)


st.caption(

    "Evaluasi Random Forest Regressor "
    "dan Linear Regression berdasarkan "
    "hasil pelatihan model."

)


st.dataframe(

    df_eval.style.format({

        "R2 Score": "{:.4f}",

        "MAE": "{:,.2f}",

        "MSE": "{:,.2f}",

        "RMSE": "{:,.2f}"

    }),

    use_container_width=True

)


# =========================================================
# 21. FEATURE IMPORTANCE
# =========================================================

st.subheader(
    "Feature Importance Random Forest"
)


st.dataframe(

    df_feature.style.format({

        "Importance": "{:.4f}"

    }),

    use_container_width=True

)


# =========================================================
# 22. GRAFIK FEATURE IMPORTANCE
# =========================================================

fig3, ax3 = plt.subplots(

    figsize=(10, 5)

)


ax3.barh(

    df_feature["Fitur"],

    df_feature["Importance"]

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
# 23. KESIMPULAN
# =========================================================

st.subheader(
    "Kesimpulan Prediksi"
)


# Nilai tertinggi Random Forest
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


# Nilai tertinggi Linear Regression
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
    f"dengan jumlah sekitar {nilai_rf:,.0f} Jumlah Balita Stunting. "
    f"\n\n"
    f"**Linear Regression:** "
    f"Prediksi tertinggi diperkirakan terjadi "
    f"di {wilayah_lr} pada tahun {tahun_lr} "
    f"dengan jumlah sekitar {nilai_lr:,.0f} Jumlah Balita Stunting."

)


# =========================================================
# 24. INFORMASI AKHIR
# =========================================================

st.markdown("---")

st.caption(

    "Sistem prediksi jumlah balita stunting "
    "Kabupaten Indramayu dan Kota Depok "
    "berdasarkan data tahun 2018–2024 "
    "dengan prediksi tahun 2025–2027."

)
