import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import sys

# Fungsi untuk membersihkan karakter yang tidak dikenal
def cleanUnknownWord(input_string):
    return re.sub(r'[^a-zA-Z\s]', '', input_string)

# Fungsi untuk preprocessing data (stem, lowercasing, menghapus stopwords)
def preprocessingData(text, stemmer, stop_words):
    text = cleanUnknownWord(text.lower())  # Ubah ke huruf kecil dan bersihkan
    words = text.split()  # Pisahkan kata-kata

    # Lakukan stemming dan hapus stopwords
    stemmed_words = [stemmer.stem(word) for word in words if word not in stop_words]

    return " ".join(stemmed_words)

# Fungsi untuk menampilkan progres
def show_progress(index, total):
    progress = (index + 1) / total * 100
    sys.stdout.write(f"\rProgress: {progress:.2f}%")
    sys.stdout.flush()

# Fungsi utama untuk membaca data dan memprosesnya
def process_data(file_path, output_file):
    # Inisialisasi stemmer dan stop words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))

    # Membaca file CSV
    data = pd.read_csv(file_path, encoding='unicode_escape')
    data['Isi Berita'] = data['Isi Berita'].fillna('').astype(str)
    data['Judul Berita'] = data['Judul Berita'].fillna('').astype(str)

    # Gabungkan 'Isi Berita' dan 'Judul Berita' menjadi satu dokumen per baris
    listDoc = data['Isi Berita'] + ' ' + data['Judul Berita']

    # Dapatkan total jumlah data untuk menghitung progres
    total_data = len(listDoc)

    # Proses setiap dokumen dan tampilkan progres
    veryClean = []
    for index, text in enumerate(listDoc):
        cleaned_text = preprocessingData(text, stemmer, stop_words)
        veryClean.append(cleaned_text)

        # Update progres setiap iterasi
        show_progress(index, total_data)

    # Menyimpan hasil yang telah diproses ke file output
    pd.Series(veryClean).to_csv(output_file, index=False)

    print("\nPreprocessing completed!")

# Path file input dan output
file = os.path.join("Dataset", "TTB", "TTB.csv")
output_file = "Dataset/TTB/TTBCleaned.csv"

# Jalankan proses
process_data(file, output_file)