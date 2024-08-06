import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Fungsi untuk melatih model
def train_model(data):
    # Memastikan kolom X1 dan X2 adalah string sebelum menggantikan tanda koma
    data['X1'] = data['X1'].astype(str).str.replace(',', '').astype(float)
    data['X2'] = data['X2'].astype(str).str.replace(',', '').astype(float)

    X = data.drop(columns=['Y'])
    y = data['Y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return knn, scaler, accuracy, report, y_test, y_pred

# Fungsi untuk melakukan prediksi
def predict(knn, scaler, input_data):
    input_data = scaler.transform([input_data])
    prediction = knn.predict(input_data)
    return prediction

# Judul aplikasi
st.write("# Red flag tender prediction app")
st.write("""###### Aplikasi ini bertujuan untuk memprediksi apakah tender tersebut berpotensi menjadi temuan
         Prediksi temuan dimaksud adalah:
         1) Keterlambatan pekerjaan
         2) Realisasi pekerjaan tidak sesuai spektek
         3) Realisasi pekerjaan tidak sesuai volume
         4) Realisasi Perusahaan pendukung pekerjaan tidak sesuai penawaran""")

st.write("###### Prediksi ini bekerja berdasarkan pengolahan data hasil audit, data umum lelang dan skor potensial fraud analisis dari ICW menggunakan algoritma machine learning KNN" )

# Mengunggah file CSV
uploaded_file = st.file_uploader("Unggah master data penelitian awal (format CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
   
    # Menampilkan jumlah baris dataset
    st.write(f"basis data penelitian awal: {data.shape[0]} record   ")
    
    knn, scaler, accuracy, report, y_test, y_pred = train_model(data)
    
    st.write(f"###### Skor akurasi Model saat ini: {accuracy}")
    #st.write("#### Overview evaluasi model:")
    #st.text(report)
    
    # Menyimpan model ke pickle object
    model_filename = 'knn_model.pkl'
    scaler_filename = 'scaler.pkl'
    
    with open(model_filename, 'wb') as file:
        pickle.dump(knn, file)
    
    with open(scaler_filename, 'wb') as file:
        pickle.dump(scaler, file)
    
    #st.write(f"Model telah disimpan ke {model_filename}")
    #st.write(f"Scaler telah disimpan ke {scaler_filename}")
    
    st.write("## Prediksi baru")
    X1 = st.text_input("Isikan nilai kontrak")
    X2 = st.text_input("Isikan nilai HPS")
    X3 = st.number_input("Isikan lama hari kalender proses lelang mulai awal pengumuman s/d penetapan pemenang akhir", value=0)
    X4 = st.number_input("Isikan rentang lama tahun anggaran pelaksanaan tender", value=0)
    
    if X1 and X2:
        X5 = float(X1.replace(',', '')) / float(X2.replace(',', ''))
        X6 = 1 - X5
    else:
        X5 = 0.0
        X6 = 0.0
    
    X7 = st.number_input("Isikan skor PFA bersumber dari https://www.opentender.net/tender", value=0.0)
    
    if st.button("Mari prediksi apakah terdapat temuan (Y) atau tidak (T)"):
        input_data = [float(X1.replace(',', '')), float(X2.replace(',', '')), X3, X4, X5, X6, X7]
        prediction = predict(knn, scaler, input_data)
        st.write(f"Hasil Prediksi: {prediction[0]}")