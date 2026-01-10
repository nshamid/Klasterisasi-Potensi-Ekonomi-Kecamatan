import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ekonomi Palembang 2025", layout="wide")

# Daftar kolom fitur yang digunakan saat training (Urutan harus SAMA persis)
fitur_ekonomi = [
    'Jumlah Penduduk', 'Kepadatan Penduduk', 'Sarana Pendidikan', 
    'Sarana Kesehatan', 'Transportasi', 'Sarana Perdagangan dan Jasa', 
    'Keberadaan Pasar dan Pertokoan', 'Bank dan Koperasi', 'IKM dan Sentra'
]

# --- 2. LOAD DATA & MODEL ---
@st.cache_resource
def load_essentials():
    # Sesuaikan path jika folder Anda berbeda
    df = pd.read_csv("Dataset/Dataset Potensi Ekonomi Kecamatan di Kota Palembang 2025.csv")
    model = joblib.load('Model/model_kmeans_potensiekonomi.pkl')
    scaler = joblib.load('Model/scaler_potensiekonomi.pkl')
    return df, model, scaler

try:
    df_raw, kmeans, scaler = load_essentials()
except Exception as e:
    st.error(f"Gagal memuat file. Pastikan folder 'data' dan 'models' sudah benar. Error: {e}")
    st.stop()

# --- 3. PREPROCESSING & LABELLING ---
# Pastikan hanya mengambil kolom numerik yang diperlukan untuk transform
X = df_raw[fitur_ekonomi]
X_scaled = scaler.transform(X)

# Prediksi Cluster
df_raw['Cluster'] = kmeans.predict(X_scaled)

# Penentuan Label Otomatis agar tidak tertukar (Rendah -> Tinggi)
# Menggunakan numeric_only=True untuk menghindari TypeError
cluster_profile = df_raw.groupby('Cluster')[fitur_ekonomi].mean()
ranking = cluster_profile.sum(axis=1).sort_values().index

mapping = {
    ranking[0]: 'Potensi Rendah',
    ranking[1]: 'Potensi Menengah',
    ranking[2]: 'Potensi Tinggi'
}
df_raw['Kategori'] = df_raw['Cluster'].map(mapping)

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b3/Logo_BPS.png", width=100)
st.sidebar.title("Menu Utama")
menu = st.sidebar.selectbox("Pilih Halaman:", ["🏠 Dashboard", "📊 Analisis Klaster", "💡 Simulasi"])

# --- 5. HALAMAN 1: DASHBOARD ---
if menu == "🏠 Dashboard":
    st.title("🏙️ Potensi Ekonomi Kecamatan Kota Palembang")
    st.markdown("Analisis Klastering menggunakan Metode **K-Means** untuk perencanaan ekonomi 2025.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Wilayah", f"{len(df_raw)} Kecamatan")
    col2.metric("Optimal Klaster", "3 Kelompok")
    col3.metric("Metode", "K-Means + PCA")

    st.write("### Dataset Utama")
    st.dataframe(df_raw, use_container_width=True)

# --- 6. HALAMAN 2: ANALISIS KLASTER ---
elif menu == "📊 Analisis Klaster":
    st.title("📊 Hasil Klasterisasi")
    
    tab1, tab2 = st.tabs(["📍 Visualisasi PCA", "📈 Profiling Indikator"])
    
    with tab1:
        st.write("### Sebaran Kecamatan (Reduksi PCA 2D)")
        pca = PCA(n_components=2)
        pca_res = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
        df_pca['Kecamatan'] = df_raw['Kecamatan']
        df_pca['Kategori'] = df_raw['Kategori']

        fig = px.scatter(
            df_pca, x='PC1', y='PC2', color='Kategori',
            hover_name='Kecamatan', text='Kecamatan',
            color_discrete_map={'Potensi Tinggi': '#2ecc71', 'Potensi Menengah': '#3498db', 'Potensi Rendah': '#e74c3c'}
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.write("### Perbandingan Rata-rata Fitur per Kategori")
        selected_feature = st.selectbox("Pilih Indikator:", fitur_ekonomi)
        
        # Groupby dengan numeric_only=True
        df_bar = df_raw.groupby('Kategori')[fitur_ekonomi].mean().reset_index()
        
        fig_bar = px.bar(
            df_bar, x='Kategori', y=selected_feature, color='Kategori',
            color_discrete_map={'Potensi Tinggi': '#2ecc71', 'Potensi Menengah': '#3498db', 'Potensi Rendah': '#e74c3c'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- 7. HALAMAN 3: SIMULASI ---
elif menu == "💡 Simulasi":
    st.title("💡 Simulasi Klasifikasi Wilayah")
    st.write("Masukkan data baru untuk mengetahui potensi ekonominya.")

    with st.form("form_simulasi"):
        cols = st.columns(2)
        inputs = {}
        for i, feat in enumerate(fitur_ekonomi):
            with cols[i % 2]:
                inputs[feat] = st.number_input(f"Masukkan {feat}", value=float(df_raw[feat].mean()))
        
        btn = st.form_submit_button("Cek Potensi")

    if btn:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        res_idx = kmeans.predict(input_scaled)[0]
        res_label = mapping[res_idx]
        
        st.success(f"### Wilayah ini termasuk dalam: **{res_label}**")
