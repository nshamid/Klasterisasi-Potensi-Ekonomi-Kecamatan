import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Sosial-Ekonomi Palembang 2025", layout="wide")

# CSS
st.markdown("""
    <style>
    /* Mengubah warna latar belakang utama */
    .stApp {
        background-color: #FFFFFF;
    }
    /* Mengubah semua teks menjadi hitam */
    h1, h2, h3, h4, h5, h6, p, li, div, span {
        color: #000000 !important;
    }
    /* Mengubah warna teks di Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    /* Mengubah warna border tabel agar terlihat jelas */
    .stDataFrame {
        border: 1px solid #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# Daftar kolom fitur asli (Sesuai model saat training)
fitur_ekonomi = [
    'Jumlah Penduduk', 'Kepadatan Penduduk', 'Sarana Pendidikan', 
    'Sarana Kesehatan', 'Transportasi', 'Sarana Perdagangan dan Jasa', 
    'Keberadaan Pasar dan Pertokoan', 'Bank dan Koperasi', 'IKM dan Sentra'
]

# --- 2. LOAD DATA & MODEL ---
@st.cache_resource
def load_essentials():
    df = pd.read_csv("Dataset/Dataset Potensi Ekonomi Kecamatan di Kota Palembang 2025.csv")
    model = joblib.load('Model/model_kmeans_potensiekonomi.pkl')
    scaler = joblib.load('Model/scaler_potensiekonomi.pkl')
    return df, model, scaler

try:
    df_raw, kmeans, scaler = load_essentials()
    df_raw['Jumlah Penduduk'] = (df_raw['Jumlah Penduduk'] * 1000).astype(int)
    df_raw['Kepadatan Penduduk'] = (df_raw['Kepadatan Penduduk'] * 1000).astype(int)
except Exception as e:
    st.error(f"Gagal memuat file. Error: {e}")
    st.stop()

# --- 3. PROSES KLASTERING ---
X = df_raw[fitur_ekonomi]
X_scaled = scaler.transform(X)
df_raw['Cluster'] = kmeans.predict(X_scaled)

# ================= PERBAIKAN UTAMA =================
cluster_score = (
    df_raw
    .groupby('Cluster')[fitur_ekonomi]
    .mean(numeric_only=True)
    .sum(axis=1)
)

sorted_cluster = cluster_score.sort_values()

# 🔒 Proteksi jika klaster tidak lengkap
if len(sorted_cluster) != 3:
    st.error(
        f"Jumlah klaster terdeteksi = {len(sorted_cluster)} (harusnya 3). "
        "Periksa preprocessing data."
    )
    st.stop()

mapping = {
    sorted_cluster.index[0]: 'Potensi Rendah',
    sorted_cluster.index[1]: 'Potensi Menengah',
    sorted_cluster.index[2]: 'Potensi Tinggi',
}

df_raw['Kategori'] = df_raw['Cluster'].map(mapping)


st.write("Distribusi Kategori:")
st.write(df_raw['Kategori'].value_counts())

# --- 4. SIDEBAR ---
st.sidebar.image("Images/logo_bps.png", width=80)
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda & Dataset", "📊 Analisis Klasterisasi", "👥 Profil Kelompok"])

st.sidebar.divider()
st.sidebar.caption("Project Magang BPS Kota Palembang")

# --- 5. HALAMAN 1: BERANDA & DATASET ---
if menu == "🏠 Beranda & Dataset":
    st.title("🏙️ Potensi Sosial-Ekonomi Kecamatan di Kota Palembang 2025")
    st.markdown("---")
    
    # Bagian 1: Informasi Project & Sumber Data
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("📌 Informasi Project")
        st.write("""
        Project ini bertujuan untuk memetakan kekuatan sosial-ekonomi wilayah di Kota Palembang menggunakan 
        pendekatan Machine Learning (**K-Means Clustering**). Analisis ini mengelompokkan kecamatan 
        ke dalam tingkatan potensi sosial-ekonomi untuk membantu perencanaan pembangunan daerah.
        """)
        st.info("""
        **Sumber Data Resmi:**
        - BPS Kota Palembang: *Kota Palembang Dalam Angka 2025*
        - BPS Kota Palembang: *Statistik Potensi Desa (Podes) Kota Palembang 2025*
        """)
    
    with col_b:
        st.subheader("📈 Statistik Data")
        st.metric("Total Wilayah", f"{len(df_raw)} Kecamatan")
        st.metric("Jumlah Indikator", f"{len(fitur_ekonomi)} Kolom")
        st.metric("Jumlah Klaster (K)", "3 Kategori")

    st.divider()

    # Bagian 2: Fitur (Atribut) Dataset 
    st.subheader("📋 Fitur (Atribut) Dataset")
    st.markdown("""
    | Fitur | Keterangan |
    | :--- | :--- |
    | **Kecamatan** | Nama kecamatan di Kota Palembang yang menjadi unit analisis |
    | **Jumlah Penduduk** | Jumlah penduduk yang berdomisili di masing-masing kecamatan |
    | **Kepadatan Penduduk** | Kepadatan penduduk per km² di setiap kecamatan |
    | **Sarana Pendidikan** | Jumlah seluruh fasilitas pendidikan (TK, SD, SMP, SMA/SMK, Perguruan Tinggi) |
    | **Sarana Kesehatan** | Jumlah seluruh fasilitas kesehatan (RS, Puskesmas, Klinik, Apotek, dll) |
    | **Sarana Perdagangan & Jasa** | Jumlah fasilitas perdagangan modern (Minimarket, Restoran, Hotel, dll) |
    | **Pasar dan Pertokoan** | Jumlah pasar tradisional dan kelompok pertokoan yang tersedia |
    | **Transportasi** | Jumlah desa/kelurahan yang memiliki akses angkutan umum dan online |
    | **Bank dan Koperasi** | Jumlah lembaga keuangan berupa bank dan koperasi |
    | **IMK dan Sentra** | Jumlah industri mikro dan kecil serta sentra industri |
    """)

    st.divider()
    
    # Bagian 3: Preview Dataset
    st.write("### 📄 Dataset Utama")
    st.dataframe(df_raw[['Kecamatan'] + fitur_ekonomi], use_container_width=True)

    # Bagian 4: Akses Project
    st.write("---")
    st.write("### 🔗 Akses Project & Dokumentasi")
    
    col_link1, col_link2 = st.columns(2)
    with col_link1:
        st.markdown("[📁 Repository GitHub Project](https://github.com/nshamid/Klasterisasi-Potensi-Ekonomi-Kecamatan/tree/main)")
    
    with col_link2:
        st.markdown("[📓 Notebook Google Colab](https://colab.research.google.com/drive/1UgG0mtfn3TkSO6zTTJIJ7U2AK2CwWvdU?usp=sharing)")

# --- 6. HALAMAN 2: ANALISIS KLASTERISASI ---
elif menu == "📊 Analisis Klasterisasi":
    st.title("📊 Hasil Analisis Klasterisasi")
    st.markdown("---")

    # Baris 1: Visualisasi PCA dan Distribusi
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("#### 📍 Peta Sebaran Klaster (PCA 2D)")
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        df_pca['Kecamatan'] = df_raw['Kecamatan']
        df_pca['Kategori'] = df_raw['Kategori']

        # PERBAIKAN WARNA: Rendah=Merah (#c0392b), Menengah=Biru (#2980b9)
        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2', color='Kategori',
            hover_name='Kecamatan', text='Kecamatan',
            color_discrete_map={'Potensi Tinggi': '#27ae60', 'Potensi Menengah': '#2980b9', 'Potensi Rendah': '#c0392b'},
            template="plotly_white"
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.write("#### 🥧 Proporsi Kategori")
        count_data = df_raw['Kategori'].value_counts().reset_index()
        # PERBAIKAN WARNA: Harus sama dengan PCA
        fig_pie = px.pie(
            count_data, names='Kategori', values='count',
            color='Kategori',
            color_discrete_map={'Potensi Tinggi': '#27ae60', 'Potensi Menengah': '#2980b9', 'Potensi Rendah': '#c0392b'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Baris 2: Daftar Kecamatan & Profiling
    st.write("### 📋 Pembagian Wilayah per Kategori")
    
    # PERBAIKAN URUTAN LIST: Menengah harus di tengah agar warna Biru cocok
    cat_cols = st.columns(3)
    kategori_list = ['Potensi Tinggi', 'Potensi Menengah', 'Potensi Rendah']
    warna_list = ['green', 'blue', 'red']

    for i, kat in enumerate(kategori_list):
        with cat_cols[i]:
            st.markdown(f"#### :{warna_list[i]}[{kat}]")
            list_kecamatan = df_raw[df_raw['Kategori'] == kat]['Kecamatan'].values
            if len(list_kecamatan) > 0:
                for kec in list_kecamatan:
                    st.write(f"- {kec}")
            else:
                st.write("*Tidak ada data*")

    st.divider()

    # Baris 3: Karakteristik Per Klaster
    st.write("### 📈 Karakteristik Indikator Per Kategori")
    feature = st.selectbox("Pilih Indikator untuk Melihat Perbandingan:", fitur_ekonomi)
    
    df_avg = df_raw.groupby('Kategori')[fitur_ekonomi].mean(numeric_only=True).reset_index()
    fig_bar = px.bar(
        df_avg, x='Kategori', y=feature, color='Kategori',
        text_auto='.2f',
        title=f"Rata-rata {feature} per Kategori",
        color_discrete_map={'Potensi Tinggi': '#27ae60', 'Potensi Menengah': '#2980b9', 'Potensi Rendah': '#c0392b'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- 7. HALAMAN 3: PROFIL KELOMPOK (CREDIT) ---
elif menu == "👥 Profil Kelompok":
    st.title("👥 Profil Kelompok Kerja Praktik")
    st.markdown("---")

    # 1. Banner Dokumentasi
    st.image("Images/banner_kelompok.jpg", 
             caption="Dokumentasi Bersama Bapak Edi Subeno, S.E., M.Si. Kepala BPS Kota Palembang", 
             use_container_width=True)
    
    st.divider()

    # 2. Logo UNSRI
    col_l1, col_l2, col_l3 = st.columns([1, 1, 1])
    with col_l2:
        st.image("Images/logo_unsri.png", width=200)
    
    # 3. Informasi Universitas (Teks Tengah)
    st.markdown("<h2 style='text-align: center;'>Teknik Informatika Bilingual 2023</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Fakultas Ilmu Komputer, Universitas Sriwijaya</h3>", unsafe_allow_html=True)
    
    st.divider()

    # 4. Informasi Akademik & Anggota
    col_info, col_anggota = st.columns(2)

    with col_info:
        st.write("### 📖 Informasi Akademik")
        st.markdown("""
        - **Mata Kuliah:** Kerja Praktik (FTI4001)
        - **Dosen Pengampu:** Yunita, S.Si., M.Cs.
        - **Dosen Pembimbing Lapangan:** Aharmisa Rahmatullah, S.ST
        """)
        
    with col_anggota:
        st.write("### 👩‍🎓 Anggota Kelompok")
        st.markdown("""
        1. **Nabilah Shamid** (09021382328147)
        2. **Saravina Zharfa Kelana P** (09021382328149)
        3. **Raka Athallah Ananta** (09021382328163)
        """)

    st.divider()
    st.success("Terima Kasih kepada BPS Kota Palembang atas bimbingan dan kesempatan yang diberikan.")
