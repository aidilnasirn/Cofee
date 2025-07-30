from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Ganti dengan kunci rahasia yang kuat

# Fungsi dummy untuk mendapatkan data training (ganti dengan koneksi DB sungguhan)
def load_training_data():
    # Ini harusnya mengambil data historis dari database Anda
    # Tambahkan kolom 'cuaca' dengan nilai kategorikal (cerah, hujan, mendung)
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
        'jumlah_hot_classic_milo': [70, 57, 85, 88, 92, 47, 80, 93, 46, 99, 31, 68, 72, 54, 37],
        'jumlah_ice_dark_chocolate': [80, 55, 48, 75, 86, 48, 38, 77, 100, 55, 75, 88, 65, 67, 32]
    }
    df = pd.DataFrame(data)
    # Lakukan One-Hot Encoding untuk kolom 'cuaca'
    df = pd.get_dummies(df, columns=['cuaca'], prefix='cuaca')
    return df

# Model KNN (akan dilatih ulang setiap kali aplikasi dimulai atau data training berubah)
knn_models = {}
# Perbarui feature_cols untuk menyertakan kolom hasil One-Hot Encoding
feature_cols = ['hari_ke', 'event_khusus', 'cuaca_cerah', 'cuaca_hujan', 'cuaca_mendung']
product_cols = ['jumlah_kopi_susu', 'jumlah_caramel_praline_macchiato', 'jumlah_ice_americano', 'jumlah_kopi_hitam', 'jumlah_latte', 'jumlah_jus_alpukat', 'jumlah_croissant', 'jumlah_hot_tea', 'jumlah_lemon_tea', 'jumlah_iced_classic_milo', 'jumlah_hot_americano', 'jumlah_expresso', 'jumlah_macha_latte', 'jumlah_ice_buttercream_tiramisu_latte', 'jumlah_hot_coppucino', 'jumlah_ice_dark_chocolate', 'jumlah_nutty_oat_latte', 'jumlah_hot_classic_milo' ] # Tambahkan semua produk Anda

def train_knn_models():
    df_train = load_training_data()
    if df_train.empty:
        print("Warning: Training data is empty. KNN models cannot be trained.")
        return

    # Pastikan semua kolom fitur yang diharapkan ada setelah One-Hot Encoding
    # Ini penting jika suatu kategori cuaca tidak muncul di data training
    for col in ['cuaca_cerah', 'cuaca_hujan', 'cuaca_mendung']:
        if col not in df_train.columns:
            df_train[col] = 0 # Tambahkan kolom dengan nilai 0 jika tidak ada

    X_train = df_train[feature_cols]
    for product in product_cols:
        y_train = df_train[product]
        # n_neighbors bisa disesuaikan, atau menggunakan cross-validation untuk optimasi
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_models[product] = knn
    print("KNN models trained successfully.")

# Panggil fungsi training saat aplikasi dimulai
train_knn_models()


# Dummy user for login
USERS = {'admin': 'admin'}

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# ... (kode sebelumnya) ...

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login_standalone.html') # Ubah di sini

# ... (kode selanjutnya) ...

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    products = {
        'kopi_susu': 'Kopi Susu',
        'caramel_praline_macchiato': 'Caramel Praline Macchiato',
        'ice_americano': 'Ice Americano',
        'latte':'Latte',
        'jus_alpukat': 'Jus Alpukat',
        'croissant': 'Croissant',
        'hot_tea': 'Hot Tea',
        'lemon_tea': 'Lemon Tea',
        'ice_classic_milo': 'Ice Classic Milo',
        'hot_americano': 'Hot Americano',
        'expresso': 'Expresso',
        'macha_latte': 'Macha Latte',
        'ice_buttercream_tiramisu_latte': 'Ice Buttercream Tiramisu Latte',
        'hot_coppucino' : 'Hot Cappucino',
        'ice_dark_chocolate': 'Ice Dark Chocolate',
        'nutty_oat_latte': 'Nutty Oat Latte',
        'hot_classic_milo': 'Hot Classic Milo'





        # Tambahkan produk lain di sini sesuai dengan product_cols
    }
    
    # Opsi cuaca untuk dropdown di HTML
    weather_options = ['cerah', 'hujan', 'mendung']

    if request.method == 'POST':
        try:
            # Input untuk kondisi prediksi
            hari_ke = int(request.form['hari_ke'])
            cuaca_input = request.form['cuaca'] # Ambil nilai cuaca dari form
            event_khusus = 1 if 'event_khusus' in request.form else 0

            # Lakukan One-Hot Encoding untuk cuaca_input
            cuaca_encoded = {'cuaca_cerah': 0, 'cuaca_hujan': 0, 'cuaca_mendung': 0}
            if f'cuaca_{cuaca_input}' in cuaca_encoded:
                cuaca_encoded[f'cuaca_{cuaca_input}'] = 1

            # Input untuk stok dan pemesanan aktual
            current_stocks = {}
            actual_orders = {}
            for key in products:
                current_stocks[key] = int(request.form.get(f'stok_{key}', 0))
                actual_orders[key] = int(request.form.get(f'pemesanan_harian_{key}', 0))


            # Buat input fitur untuk prediksi sesuai dengan feature_cols
            input_features = np.array([[
                hari_ke,
                event_khusus,
                cuaca_encoded['cuaca_cerah'],
                cuaca_encoded['cuaca_hujan'],
                cuaca_encoded['cuaca_mendung']
            ]])
            
            predictions = {}

            # Lakukan prediksi untuk setiap produk
            for product_key, product_name in products.items():
                if f'jumlah_{product_key}' in knn_models:
                    pred = knn_models[f'jumlah_{product_key}'].predict(input_features)[0]
                    predictions[product_key] = max(0, round(pred)) # Pastikan tidak ada prediksi negatif
                else:
                    predictions[product_key] = 0 # Produk tidak ada modelnya

            # Hitung kekurangan dan rekomendasi pemesanan
            results = []
            for product_key, product_name in products.items():
                predicted_demand = predictions.get(product_key, 0)
                current_stock = current_stocks.get(product_key, 0)
                actual_order = actual_orders.get(product_key, 0)

                # Kekurangan berdasarkan stok saat ini dan permintaan yang diprediksi
                kekurangan_prediksi = max(0, predicted_demand - current_stock)

                # Rekomendasi pemesanan (bisa disesuaikan logikanya)
                # Misalnya, kita ingin stok mencukupi setidaknya 110% dari prediksi permintaan
                rekomendasi_pemesanan = max(0, int(predicted_demand * 1.1) - current_stock) # Contoh: target 110% dari prediksi

                results.append({
                    'product_name': product_name,
                    'predicted_demand': predicted_demand,
                    'current_stock': current_stock,
                    'actual_order': actual_order,
                    'kekurangan_prediksi': kekurangan_prediksi,
                    'rekomendasi_pemesanan': rekomendasi_pemesanan
                })

            session['prediction_results'] = results
            return redirect(url_for('result'))

        except ValueError as e:
            flash(f'Invalid input: {e}. Please ensure all numerical fields are filled correctly.', 'danger')
        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')
    return render_template('predict.html', products=products, weather_options=weather_options)

@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    results = session.pop('prediction_results', []) # Ambil dan hapus dari session
    return render_template('result.html', results=results)


if __name__ == '__main__':
    app.run(debug=True) # Set debug=False untuk produksi