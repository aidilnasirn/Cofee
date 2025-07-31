import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Stok Kopi", layout="wide")

# --- DATA & MODEL TRAINING ---
@st.cache_data
def load_training_data():
    """
    Memuat dan melakukan pra-pemrosesan data training.
    """
    data = {
        'hari_ke': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'cuaca': ['cerah', 'cerah', 'hujan', 'cerah', 'mendung', 'cerah', 'hujan', 'cerah', 'mendung', 'cerah', 'cerah', 'hujan', 'cerah', 'mendung', 'cerah'],
        'event_khusus': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        'jumlah_kopi_susu': [80, 90, 120, 95, 85, 92, 110, 88, 93, 100, 82, 95, 115, 89, 94],
        'jumlah_caramel_praline_macchiato': [60, 65, 80, 70, 62, 68, 75, 63, 67, 72, 61, 69, 78, 64, 66],
        'jumlah_ice_americano': [40, 45, 55, 48, 42, 47, 50, 43, 46, 49, 41, 48, 52, 44, 47],
        'jumlah_kopi_hitam': [40, 55, 88, 75, 56, 78, 68, 57, 68, 57, 55, 78, 35, 57, 92],
        'jumlah_latte': [83, 95, 100, 97, 83, 93, 80, 88, 93, 110, 82, 95, 115, 89, 94],
        'jumlah_jus_alpukat': [65, 77, 82, 67, 69, 65, 75, 45, 67, 74, 61, 69, 78, 64, 66],
        'jumlah_croissant': [40, 47, 65, 68, 72, 47, 60, 73, 86, 39, 71, 88, 92, 64, 47],
        'jumlah_hot_tea': [70, 65, 88, 95, 56, 68, 78, 87, 100, 35, 75, 88, 65, 27, 62],
        'jumlah_lemon_tea': [60, 80, 130, 65, 25, 82, 110, 88, 93, 100, 82, 95, 115, 89, 74],
        'jumlah_iced_classic_milo': [80, 75, 60, 80, 92, 38, 65, 43, 27, 92, 21, 59, 78, 64, 66],
        'jumlah_hot_americano': [60, 45, 25, 68, 72, 87, 50, 83, 46, 49, 41, 48, 53, 44, 47],
        'jumlah_expresso': [60, 75, 78, 85, 96, 68, 48, 37, 78, 87, 75, 38, 85, 97, 92],
        'jumlah_macha_latte': [93, 55, 110, 37, 73, 83, 80, 58, 33, 110, 72, 45, 115, 59, 94],
        'jumlah_ice_buttercream_tiramisu_latte': [75, 87, 62, 57, 89, 55, 35, 75, 87, 94, 81, 49, 48, 34, 96],
        'jumlah_hot_coppucino': [80, 57, 75, 88, 42, 37, 80, 43, 86, 49, 81, 98, 32, 74, 97],
        'jumlah_ice_dark_chocolate': [80, 55, 48, 75, 86, 48, 38, 77, 100, 55, 75, 88, 65, 67, 32],
        'jumlah_nutty_oat_latte': [85, 67, 42, 37, 79, 85, 45, 35, 37, 74, 81, 99, 28, 74, 26],
        'jumlah_hot_classic_milo': [70, 57, 85, 88, 92, 47, 80, 93, 46, 99, 31, 68, 72, 54, 37]
    }
    df = pd.DataFrame(data)
    df = pd.get_dummies(df, columns=['cuaca'], prefix='cuaca')
    return df

@st.cache_resource
def train_knn_models():
    """
    Melatih model KNN untuk setiap produk berdasarkan data training.
    """
    df_train = load_training_data()
    feature_cols = ['hari_ke', 'event_khusus', 'cuaca_cerah', 'cuaca_hujan', 'cuaca_mendung']
    product_cols = [col for col in df_train.columns if col.startswith('jumlah_')]
    knn_models = {}
    for col in ['cuaca_cerah', 'cuaca_hujan', 'cuaca_mendung']:
        if col not in df_train.columns:
            df_train[col] = 0
    X_train = df_train[feature_cols]
    for product in product_cols:
        y_train = df_train[product]
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train, y_train)
        knn_models[product] = knn
    print("KNN models trained successfully.")
    return knn_models

# --- FUNGSI CSS ---
def local_css(file_name):
    """Fungsi untuk membaca file CSS lokal dan menyisipkannya."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- MAIN APP LOGIC ---
knn_models = train_knn_models()
df_train = load_training_data()
products_map = {
    'jumlah_kopi_susu': 'Kopi Susu', 'jumlah_caramel_praline_macchiato': 'Caramel Praline Macchiato',
    'jumlah_ice_americano': 'Ice Americano', 'jumlah_kopi_hitam': 'Kopi Hitam', 'jumlah_latte': 'Latte',
    'jumlah_jus_alpukat': 'Jus Alpukat', 'jumlah_croissant': 'Croissant', 'jumlah_hot_tea': 'Hot Tea',
    'jumlah_lemon_tea': 'Lemon Tea', 'jumlah_iced_classic_milo': 'Ice Classic Milo',
    'jumlah_hot_americano': 'Hot Americano', 'jumlah_expresso': 'Expresso', 'jumlah_macha_latte': 'Macha Latte',
    'jumlah_ice_buttercream_tiramisu_latte': 'Ice Buttercream Tiramisu Latte', 'jumlah_hot_coppucino': 'Hot Cappucino',
    'jumlah_ice_dark_chocolate': 'Ice Dark Chocolate', 'jumlah_nutty_oat_latte': 'Nutty Oat Latte',
    'jumlah_hot_classic_milo': 'Hot Classic Milo'
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- UI COMPONENTS ---
def show_login_page():
    """Menampilkan halaman login."""
    local_css("style.css") # Terapkan CSS juga di halaman login
    st.title("Login Sistem Prediksi OT Coffee")
    with st.form("login_form"):
        username = st.text_input("Username", value="Masukkan username")
        password = st.text_input("Password", type="password", value="Masukkan password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Username atau password salah!")

def show_main_app():
    """Menampilkan aplikasi utama setelah login berhasil."""
    # **PANGGIL FUNGSI CSS DI SINI**
    local_css("style.css")
    
    with st.sidebar:
        st.header(f"Selamat Datang, Admin!")
        st.info("Aplikasi ini menggunakan KNN untuk memprediksi permintaan produk.")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.results_df = None
            st.rerun()

    st.title("Dashboard Prediksi & Rekomendasi Stok")
    st.markdown("---")

    with st.form("prediction_form"):
        st.header("1. Kondisi Prediksi")
        col1, col2, col3 = st.columns(3)
        with col1:
            hari_ke = st.number_input("Prediksi untuk Hari ke-", min_value=1, value=16)
        with col2:
            cuaca_input = st.selectbox("Kondisi Cuaca", options=['cerah', 'hujan', 'mendung'])
        with col3:
            event_khusus = st.checkbox("Ada Event Khusus?")

        st.markdown("---")
        st.header("2. Input Stok & Pemesanan Aktual (Opsional)")
        for p_col, p_name in products_map.items():
            cols = st.columns([0.5, 0.25, 0.25])
            with cols[0]:
                st.write(p_name)
            with cols[1]:
                st.number_input("Stok Saat Ini", key=f"stok_{p_col}", min_value=0, value=0, label_visibility="collapsed")
            with cols[2]:
                st.number_input("Pemesanan Aktual", key=f"pemesanan_{p_col}", min_value=0, value=0, label_visibility="collapsed")
        
        submitted = st.form_submit_button("Buat Prediksi & Rekomendasi")

    if submitted:
        cuaca_encoded = {'cuaca_cerah': 0, 'cuaca_hujan': 0, 'cuaca_mendung': 0}
        cuaca_encoded[f'cuaca_{cuaca_input}'] = 1
        event_val = 1 if event_khusus else 0

        input_features = np.array([[
            hari_ke, event_val, cuaca_encoded['cuaca_cerah'],
            cuaca_encoded['cuaca_hujan'], cuaca_encoded['cuaca_mendung']
        ]])
        
        results = []
        for p_col, p_name in products_map.items():
            pred = knn_models[p_col].predict(input_features)[0]
            predicted_demand = max(0, round(pred))
            current_stock = st.session_state[f'stok_{p_col}']
            rekomendasi_pemesanan = max(0, int(predicted_demand * 1.1) - current_stock)
            results.append({
                'Nama Produk': p_name,
                'Prediksi Permintaan': predicted_demand,
                'Stok Saat Ini': current_stock,
                'Rekomendasi Pemesanan': rekomendasi_pemesanan
            })
        st.session_state.results_df = pd.DataFrame(results)

    if st.session_state.results_df is not None:
        st.markdown("---")
        st.header("Hasil Prediksi")
        st.dataframe(st.session_state.results_df.style.highlight_max(subset=['Prediksi Permintaan', 'Rekomendasi Pemesanan'], color='#e94560', axis=0))

# --- KONTROL ALUR APLIKASI ---
if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
