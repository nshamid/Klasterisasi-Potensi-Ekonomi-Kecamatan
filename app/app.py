import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.decomposition import PCA

# --- CONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Ekonomi Palembang", layout="wide")

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_essentials():
    df = pd.read_csv("Dataset Potensi Ekonomi Kecamatan di Kota Palembang 2025.csv")
    model = joblib.load('model_kmeans_potensiekonomi.pkl')
    scaler = joblib.load('scaler_potensiekonomi.pkl')
    return df, model, scaler

try:
    df_raw, kmeans, scaler = load_essentials()
except Exception as e:
    st.error(f"Gagal memuat file. Pastikan file .csv dan .pkl ada di repository. Error: {e}")
    st.stop()

# --- PREPROCESSING UNTUK DASHBOARD ---
X = df_raw.drop(columns=['Kecamatan'])
X_scaled = scaler.transform(X)
df_raw['Cluster'] = kmeans.predict(X_scaled)

# Mapping Label (Sesuaikan dengan hasil analisis Anda di Colab)
# Di sini saya asumsikan urutan berdasarkan rata-rata fitur ekonomi
profiling = df_raw.drop(columns='Kecamatan').groupby('Cluster').mean().sum(axis=1).sort_values()
mapping = {profiling.index[0]: 'Potensi Rendah', 
           profiling.index[1]: 'Potensi Menengah', 
           profiling.index[2]: 'Potensi Tinggi'}
df_raw['Kategori'] = df_raw['Cluster'].map(mapping)

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b3/Logo_BPS.png", width=100)
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda & Data", "📊 Analisis Klaster", "💡 Simulasi Prediksi"])

st.sidebar.info("Project Magang BPS Kota Palembang 2025")

# --- HALAMAN 1: BERANDA ---
if menu == "🏠 Beranda & Data":
    st.title("🏙️ Analisis Klaster Potensi Ekonomi Kecamatan")
    st.subheader("Kota Palembang Tahun 2025")
    
    st.write("""
    Dashboard ini digunakan untuk mengelompokkan kecamatan di Kota Palembang berdasarkan 
    9 indikator ekonomi utama menggunakan algoritma **K-Means Clustering**.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Kecamatan", len(df_raw))
    col2.metric("Jumlah Klaster", "3 Kategori")
    col3.metric("Sumber Data", "BPS Palembang")

    st.divider()
    st.write("### Dataset Potensi Ekonomi")
    st.dataframe(df_raw, use_container_width=True)

# --- HALAMAN 2: ANALISIS KLASTER ---
elif menu == "📊 Analisis Klaster":
    st.title("📊 Hasil Analisis Klastering")
    
    tab1, tab2 = st.tabs(["📈 Visualisasi Sebaran (PCA)", "📋 Profiling Kategori"])
    
    with tab1:
        st.write("### Peta Sebaran Dimensi Ekonomi (PCA)")
        # Reduksi dimensi untuk visualisasi
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        df_pca['Kecamatan'] = df_raw['Kecamatan']
        df_pca['Kategori'] = df_raw['Kategori']

        fig_pca = px.scatter(
            df_pca, x='PC1', y='PC2', color='Kategori',
            hover_name='Kecamatan', text='Kecamatan',
            color_discrete_map={'Potensi Tinggi': 'green', 'Potensi Menengah': 'blue', 'Potensi Rendah': 'red'},
            title="Sebaran Kecamatan Berdasarkan Kemiripan Karakteristik"
        )
        fig_pca.update_traces(textposition='top center')
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("Keterangan: Semakin dekat posisi dua kecamatan, semakin mirip potensi ekonominya.")

    with tab2:
        st.write("### Rata-rata Indikator Per Kategori")
        df_profile = df_raw.drop(columns='Kecamatan').groupby('Kategori').mean().reset_index()
        
        # Pilih fitur untuk ditampilkan di bar chart
        feature_to_plot = st.selectbox("Pilih Indikator untuk Dibandingkan:", X.columns)
        
        fig_bar = px.bar(
            df_profile, x='Kategori', y=feature_to_plot,
            color='Kategori',
            title=f"Perbandingan Rata-rata {feature_to_plot}",
            color_discrete_map={'Potensi Tinggi': 'green', 'Potensi Menengah': 'blue', 'Potensi Rendah': 'red'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# --- HALAMAN 3: SIMULASI PREDIKSI ---
elif menu == "💡 Simulasi Prediksi":
    st.title("💡 Simulasi Penentuan Klaster")
    st.write("Masukkan data indikator ekonomi untuk memprediksi kategori potensi wilayah tersebut.")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            jml_penduduk = st.number_input("Jumlah Penduduk", value=100000)
            kepadatan = st.number_input("Kepadatan Penduduk", value=5000)
            pendidikan = st.number_input("Jumlah Sarana Pendidikan", value=20)
            kesehatan = st.number_input("Jumlah Sarana Kesehatan", value=15)
            transp = st.number_input("Jumlah Transportasi", value=10)
            
        with col2:
            dagang = st.number_input("Sarana Perdagangan & Jasa", value=15)
            pasar = st.number_input("Keberadaan Pasar/Toko", value=5)
            bank = st.number_input("Jumlah Bank & Koperasi", value=5)
            ikm = st.number_input("Jumlah IKM & Sentra", value=5)
            
        submitted = st.form_submit_button("Prediksi Kategori")
        
    if submitted:
        # Satukan input menjadi array
        new_data = np.array([[jml_penduduk, kepadatan, pendidikan, kesehatan, 
                              transp, dagang, pasar, bank, ikm]])
        
        # Scaling & Predict
        new_data_scaled = scaler.transform(new_data)
        prediction = kmeans.predict(new_data_scaled)[0]
        hasil_kategori = mapping[prediction]
        
        st.divider()
        if "Tinggi" in hasil_kategori:
            st.success(f"### Hasil: **{hasil_kategori}**")
        elif "Menengah" in hasil_kategori:
            st.info(f"### Hasil: **{hasil_kategori}**")
        else:
            st.warning(f"### Hasil: **{hasil_kategori}**")
