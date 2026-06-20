import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import accuracy_score
from PIL import Image

st.set_page_config(page_title="Age-Invariant Face Recognition", layout="wide")

st.title("Sistem Deteksi Kemiripan Wajah Lintas Usia (PCA)")
st.write("Menggunakan algoritma Principal Component Analysis (Eigenfaces) dengan deteksi wajah Haar Cascade.")

# Standar ukuran gambar sesuai panduan PDF
IMG_SIZE = (100, 100) 
DATASET_PATH = "dataset"

def detect_and_crop_face(img_array):
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None
        
    x, y, w, h = faces[0]
    face_crop = gray[y:y+h, x:x+w]
    
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return face_normalized.flatten()

@st.cache_data
def load_and_split_dataset():
    X_train, X_test, y_train, y_test = [], [], [], []
    if not os.path.exists(DATASET_PATH):
        return None, None, None, None
        
    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder): continue
            
        person_vectors = []
        # Menggunakan sorted agar file diurutkan berdasarkan nama (001A02.jpg akan dibaca lebih dulu)
        for filename in sorted(os.listdir(person_folder)):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_folder, filename)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                vector = detect_and_crop_face(img)
                if vector is not None:
                    person_vectors.append(vector)
                    
        # Trik Pemisahan Dinamis: Index 0 (foto termuda) jadi data uji, sisanya data latih
        if len(person_vectors) >= 2:
            X_test.append(person_vectors[0])
            y_test.append(person_name)
            
            for vec in person_vectors[1:]:
                X_train.append(vec)
                y_train.append(person_name)
                    
    if len(X_train) == 0 or len(X_test) == 0: return None, None, None, None
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

# Menjalankan pemuatan dataset
X_train, X_test, y_train, y_test = load_and_split_dataset()

tab1, tab2 = st.tabs(["EDA & Laporan Model (Data Latih/Uji)", "Demo Uji Kemiripan (Foto Baru)"])

with tab1:
    st.header("Exploratory Data Analysis & Evaluasi PCA")
    if X_train is None:
        st.warning("Folder 'dataset' tidak ditemukan atau kosong. Silakan buat folder 'dataset' dan isi dengan gambar wajah.")
    else:
        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Data Latih (Foto Dewasa)", f"{len(X_train)} gambar")
        col2.metric("Data Uji (Foto Masa Kecil)", f"{len(X_test)} gambar")
        col3.metric("Total Explained Variance", f"{np.sum(pca.explained_variance_ratio_)*100:.2f}%")
        
        st.subheader("Evaluasi Akurasi (>50% Target)")
        
        cosine_sim_matrix = cosine_similarity(X_test_pca, X_train_pca)
        y_pred_cosine = [y_train[np.argmax(sim)] for sim in cosine_sim_matrix]
        acc_cosine = accuracy_score(y_test, y_pred_cosine) * 100
        
        euclidean_dist_matrix = euclidean_distances(X_test_pca, X_train_pca)
        y_pred_euclidean = [y_train[np.argmin(dist)] for dist in euclidean_dist_matrix]
        acc_euclidean = accuracy_score(y_test, y_pred_euclidean) * 100
        
        st.write(f"Akurasi Cosine Similarity: {acc_cosine:.2f}%")
        st.write(f"Akurasi Euclidean Distance: {acc_euclidean:.2f}%")
        
        st.subheader("Visualisasi Mean Face")
        mean_face = pca.mean_.reshape(IMG_SIZE)
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(mean_face, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

with tab2:
    st.header("Uji Kemiripan Wajah Kustom")
    st.write("Unggah foto masa kecil dan foto dewasa untuk dihitung tingkat kemiripannya.")
    
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        upload_kecil = st.file_uploader("Unggah Foto Masa Kecil", type=['jpg', 'jpeg', 'png'])
    with col_img2:
        upload_dewasa = st.file_uploader("Unggah Foto Dewasa", type=['jpg', 'jpeg', 'png'])
        
    if upload_kecil and upload_dewasa and X_train is not None:
        img_kecil = np.array(Image.open(upload_kecil).convert('RGB'))
        img_dewasa = np.array(Image.open(upload_dewasa).convert('RGB'))
        
        vec_kecil = detect_and_crop_face(img_kecil)
        vec_dewasa = detect_and_crop_face(img_dewasa)
        
        if vec_kecil is None or vec_dewasa is None:
            st.error("Wajah tidak terdeteksi pada salah satu atau kedua foto. Coba foto yang lebih jelas!")
        else:
            pca_kecil = pca.transform(vec_kecil.reshape(1, -1))
            pca_dewasa = pca.transform(vec_dewasa.reshape(1, -1))
            
            cos_sim = cosine_similarity(pca_kecil, pca_dewasa)[0][0]
            euc_dist = euclidean_distances(pca_kecil, pca_dewasa)[0][0]
            
            st.markdown("---")
            st.subheader("Hasil Perhitungan Matematis")
            
            # Penetapan nilai batas kemiripan
            threshold_cosine = 0.80
            threshold_euclidean = 15.0
            
            st.write(f"1. Metode Cosine Similarity: {cos_sim:.4f}")
            if cos_sim >= threshold_cosine:
                st.success("Kesimpulan Cosine: MIRIP (Arah vektor sejajar)")
            else:
                st.error("Kesimpulan Cosine: TIDAK MIRIP")
                
            st.write(f"2. Metode Euclidean Distance: {euc_dist:.4f}")
            if euc_dist < threshold_euclidean:
                st.success("Kesimpulan Euclidean: MIRIP (Jarak ruang dekat)")
            else:
                st.error("Kesimpulan Euclidean: TIDAK MIRIP")
