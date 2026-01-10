import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Ekonomi Palembang 2025", layout="wide")

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
except Exception as e:
    st.error(f"Gagal memuat file. Error: {e}")
    st.stop()

# --- 3. PROSES KLASTERING ---
X = df_raw[fitur_ekonomi]
X_scaled = scaler.transform(X)
df_raw['Cluster'] = kmeans.predict(X_scaled)

# Penentuan Label Otomatis (Rendah -> Tinggi)
ranking = df_raw.groupby('Cluster')[fitur_ekonomi].mean(numeric_only=True).sum(axis=1).sort_values().index
mapping = {ranking[0]: 'Potensi Rendah', ranking[1]: 'Potensi Menengah', ranking[2]: 'Potensi Tinggi'}
df_raw['Kategori'] = df_raw['Cluster'].map(mapping)

# --- 4. SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b3/Logo_BPS.png", width=80)
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda & Dataset", "📊 Analisis Klasterisasi"])

st.sidebar.divider()
st.sidebar.caption("Project Magang BPS Kota Palembang 2025")

# --- 5. HALAMAN 1: BERANDA & DATASET ---
if menu == "🏠 Beranda & Dataset":
    st.title("🏙️ Potensi Ekonomi Kecamatan Kota Palembang 2025")
    st.markdown("---")
    
    # Sumber Data & Info
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Informasi Project")
        st.write("""
        Project ini bertujuan untuk memetakan kekuatan ekonomi wilayah di Kota Palembang menggunakan 
        pendekatan Machine Learning (K-Means Clustering). Hasil dari analisis ini diharapkan dapat 
        menjadi bahan pertimbangan dalam pemerataan pembangunan ekonomi wilayah.
        """)
        st.info("""
        **Sumber Data:**
        - Publikasi BPS: *Kota Palembang Dalam Angka 2025*
        - Publikasi BPS: *Statistik Potensi Desa Kota Palembang 2025*
        """)
    
    with col_b:
        st.subheader("Statistik Ringkas")
        st.metric("Total Wilayah", f"{len(df_raw)} Kecamatan")
        st.metric("Indikator Digunakan", f"{len(fitur_ekonomi)} Kolom")

    st.divider()
    
    # Eksplorasi Data
    st.write("### Preview Dataset Utama")
    st.dataframe(df_raw[['Kecamatan'] + fitur_ekonomi], use_container_width=True)
    
    st.write("### Statistik Deskriptif")
    st.table(df_raw[fitur_ekonomi].describe().T)

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
        fig_pie = px.pie(
            count_data, names='Kategori', values='count',
            color='Kategori',
            color_discrete_map={'Potensi Tinggi': '#27ae60', 'Potensi Menengah': '#2980b9', 'Potensi Rendah': '#c0392b'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Baris 2: Daftar Kecamatan & Profiling
    st.write("### 📋 Pembagian Wilayah per Kategori")
    
    # Menampilkan daftar kecamatan dalam kolom agar rapi
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
