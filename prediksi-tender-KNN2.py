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
st.write("# Aplikasi prediksi pekerjaan yang berpotensi bermasalah (temuan)")
st.write("##### Implementasi pemograman machine learning dengan algoritma KNN")

# Mengunggah file CSV
uploaded_file = st.file_uploader("Unggah database file (format CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Tampilan database yang diunggah (5 record teratas):")
    st.write(data.head())
    
    # Menampilkan jumlah baris dataset
    st.write(f"Jumlah record database: {data.shape[0]}")
    
    knn, scaler, accuracy, report, y_test, y_pred = train_model(data)
    
    st.write(f"###### Skor akurasi Model: {accuracy}")
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
    
    # Menampilkan confusion matrix
    st.write("###### Tampilan Confusion Matrix Chart (test size 0.2)")
    f, ax = plt.subplots(figsize=(5, 2))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", ax=ax)
    plt.xlabel("prediksi")
    plt.ylabel("aktual")
    st.pyplot(f)

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