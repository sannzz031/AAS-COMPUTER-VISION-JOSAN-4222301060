OCR PLAT NOMOR KENDARAAN MENGGUNAKAN VISUAL LANGUAGE MODEL (VLM)

Mata Kuliah    : [Computer Vision] 
Nama Mahasiswa : Josan Mauritz Sharon Nunuhitu 
Nim            : 4222301060
Tanggal        : [26/07/2026]

BAB 1: PENDAHULUAN

1.1	Latar Belakang
Optical Character Recognition (OCR) adalah teknologi yang memungkinkan konversi teks dari gambar menjadi teks digital yang dapat dibaca dan diproses oleh komputer. Dalam beberapa tahun terakhir, perkembangan model kecerdasan buatan, khususnya Visual Language Model (VLM), membuka peluang baru untuk melakukan OCR dengan pendekatan yang lebih alami dan kontekstual.
VLM adalah model AI yang mampu memahami dan memproses baik gambar (visual) maupun teks secara bersamaan. Berbeda dengan OCR tradisional yang hanya mendeteksi pola karakter, VLM dapat "membaca" gambar secara holistik dan memahami konteks visual dari teks yang muncul. Hal ini membuat VLM sangat cocok untuk tugas-tugas seperti membaca plat nomor kendaraan yang memiliki variasi font, ukuran, dan kondisi pencahayaan yang beragam.
Dalam proyek ini, kami mengimplementasikan sistem OCR plat nomor menggunakan VLM yang dijalankan melalui LM Studio dan diintegrasikan dengan Python. Tujuan utama adalah mengembangkan sistem yang mampu membaca plat nomor kendaraan dari gambar dan mengevaluasi akurasi prediksi menggunakan metrik Character Error Rate (CER).

1.2	Tujuan
Tujuan dari proyek ini adalah:
1.	Mengimplementasikan sistem OCR plat nomor kendaraan menggunakan Visual Language Model (VLM).
2.	Mengintegrasikan LM Studio sebagai inference server dengan bahasa pemrograman Python.
3.	Mengevaluasi performa model menggunakan metrik Character Error Rate (CER).

4.	Menyimpan hasil prediksi dalam format CSV untuk analisis lebih lanjut.

1.3	Ruang Lingkup
Proyek ini mencakup:
•	Dataset plat nomor kendaraan Indonesia (100 gambar dengan ground truth).

•	Penggunaan VLM melalui LM Studio (menggunakan model qwen2-vl-2b-instruct).

•	Proses cropping bounding box dari label yang tersedia.

•	Inferensi model terhadap plat nomor yang sudah di-crop.

•	Perhitungan CER untuk setiap prediksi.

•	Export hasil ke CSV dan summary file.


BAB 2: KONSEP DAN TEORI

2.1	Visual Language Model (VLM)
Visual Language Model adalah model kecerdasan buatan multimodal yang mampu memproses dan memahami informasi dari dua modalitas berbeda: gambar (visual) dan teks. VLM dilatih pada dataset besar yang berisi pasangan gambar-teks, sehingga model mampu:
•	Memahami konten visual: Mengenali objek, teks, dan konteks dalam gambar.

•	Menghubungkan visual dengan bahasa: Menghasilkan deskripsi, jawaban, atau interpretasi teks berdasarkan input visual.
•	Melakukan reasoning visual: Menjawab pertanyaan kompleks tentang konten gambar. VLM berbeda dengan model OCR tradisional karena:
•	OCR tradisional hanya mendeteksi karakter berdasarkan pola piksel.

•	VLM memahami konteks keseluruhan gambar dan dapat menginterpretasikan teks dalam konteks visualnya.
Contoh arsitektur VLM populer:
•	LLaVA (Large Language and Vision Assistant)

•	Qwen-VL (Qwen Vision Language)

•	BakLLaVA

2.2	LM Studio
LM Studio adalah aplikasi desktop yang memungkinkan pengguna untuk menjalankan model bahasa besar (LLM) dan VLM secara lokal tanpa perlu koneksi internet. Fitur utama LM Studio:
•	Model Management: Download dan kelola berbagai model AI.

•	Local Server: Menyediakan API server yang kompatibel dengan OpenAI API.

•	Inference: Menjalankan inferensi model secara lokal dengan GPU atau CPU.

•	User-friendly Interface: GUI yang memudahkan pengguna non-teknis.
LM Studio menggunakan endpoint /v1/chat/completions yang kompatibel dengan OpenAI API, sehingga memudahkan integrasi dengan Python menggunakan library requests.

2.3	Character Error Rate (CER)
Character Error Rate (CER) adalah metrik evaluasi yang umum digunakan dalam tugas OCR dan speech recognition. CER mengukur kesalahan karakter antara prediksi model dan ground truth.

Rumus CER:

Dimana:
•	S (Substitutions): Jumlah karakter yang salah substitusi (diganti dengan karakter yang salah).
•	D (Deletions): Jumlah karakter yang dihapus (hilang dalam prediksi).

•	I (Insertions): Jumlah karakter yang disisipkan (karakter tambahan dalam prediksi).

•	N: Jumlah karakter pada ground truth.

Contoh Perhitungan:

Ground Truth	Prediction	S	D	I	CER
B2134PZJ	B2136PZJ	1	0	0	1/8 = 0.125
B2134PZJ	82136PZJ	2	0	0	2/8 = 0.25
B2134PZJ	B2134PZ	0	1	0	1/8 = 0.125

Interpretasi CER:
•	CER = 0: Prediksi sempurna (tidak ada kesalahan).

•	CER = 0.1: 10% karakter salah (masih dapat diterima).

•	CER > 0.3: Akurasi buruk (tidak dapat diandalkan).

BAB 3: METODOLOGI

3.1	Dataset
Dataset yang digunakan adalah "Indonesian License Plate Dataset" yang terdiri dari:
•	100 gambar plat nomor kendaraan Indonesia.

•	Labels dalam format YOLO dengan bounding box dan ground truth teks plat.

Struktur folder:


Format label:

Contoh: 0 0.5 0.5 0.2 0.1 B1234XYZ

3.2	Alur Sistem
Berikut adalah alur sistem yang diimplementasikan:

1.	Load Dataset
Memuat gambar, bounding box, dan ground truth dari dataset.

2.	Preprocessing
Melakukan crop bounding box, padding, dan enhancement pada gambar plat nomor.

3.	Inferensi
Mengirim gambar hasil crop ke LM Studio API (http://127.0.0.1:1234/v1/chat/completions) dengan prompt pembacaan plat nomor, menggunakan model qwen2-vl-2b-instruct.

4.	Post-processing
Membersihkan hasil prediksi, mengekstrak teks plat, dan melakukan koreksi format.

5.	Evaluasi
Menghitung CER, mengekspor hasil ke CSV, dan membuat summary.

3.3	Implementasi Teknis

3.3.1	Koneksi ke LM Studio


3.3.2	Cropping Bounding Box


 
3.3.3	Image Enhancement


3.3.4	Inferensi API Call


3.3.5	Perhitungan CER


3.4	Model yang Digunakan
Dalam implementasi ini, kami menggunakan model:
•	Model: qwen2-vl-2b-instruct

•	Parameter: 2.2 billion (2.2B)

•	Keunggulan: Mendukung input multimodal (gambar + teks) melalui LM Studio. Alasan Pemilihan:
•	Kompatibel dengan LM Studio dan OpenAI API format.

•	Ukuran relatif kecil (2.2B) sehingga dapat berjalan di hardware terbatas.

•	Mendukung vision input dengan format base64.


BAB 4: HASIL DAN ANALISIS

4.1	Ringkasan Performa

Metrik	Nilai
Total plat diproses	197
Rata-rata CER	0.0947
Exact match (CER = 0)	137 / 197
Akurasi exact match	69.54%

Metrik	Nilai
Model	qwen2-vl-2b-instruct

4.2	Distribusi CER

Rentang CER	Jumlah Plat	Persentase
CER = 0 (sempurna)	137	69.54%
0 < CER ≤ 0.15	25	12.69%
0.15 < CER ≤ 0.30	18	9.14%
0.30 < CER ≤ 0.50	10	5.08%
CER > 0.50	7	3.55%

Sebagian besar kesalahan terjadi pada tingkat substitusi tunggal (CER sekitar 0.125–0.143), yang menunjukkan bahwa model cukup baik namun masih kesulitan membedakan karakter yang mirip secara visual.

4.3	Contoh Kasus Sukses (CER = 0)

Image	Ground Truth	Prediction
test002_0	BG1352AE	BG1352AE
test003_0	B2634UZF	B2634UZF
test006_0	T1329KC	T1329KC
test007_0	AD8865EE	AD8865EE
test008_0	DK1157AAB	DK1157AAB
test010_0	B9416PCN	B9416PCN
test015_0	AD1798BT	AD1798BT
test024_0	B2964TRU	B2964TRU
test040_0	S1595LL	S1595LL
test099_1	B1892WZT	B1892WZT

Image	Ground Truth	Prediction
test100_1	B2134PZJ	B2134PZJ

Model berhasil membaca plat dengan format standar (huruf–angka–huruf) tanpa kesalahan, termasuk plat dengan kombinasi huruf ganda (AD, BG, DK) dan angka yang panjang.

4.4	Contoh Kasus Gagal (CER > 0)

Image	Ground Truth	Prediction	CER	Jenis Kesalahan
test001_0	B9140BCD	B5140BCD	0.125	Substitusi (9→5)
test005_0	DD8798KM	DD2798KM	0.125	Substitusi (8→2)
test013_1	B1128WOS	B128W	0.375	Delesi (1,1,O,S?) + substitusi
test016_0	E1517DQ	E1517	0.286	Delesi 2 karakter (D, Q)

test022_2	
L3823IB	
O234XYZ	
1.000	Semua karakter salah (total failure)
test033_3	L1762ABJ	B1234XYZ	0.875	Hampir semua salah
test075_2	7066OF	O00UP	0.833	Substitusi & delesi masif
test080_1	B1861TOI	B1961TQI	0.250	Substitusi (8→9, O→Q)

Analisis Kesalahan Umum:
5.	Substitusi karakter mirip – sering terjadi pada pasangan:

◦	9 dan 5 (test001_0)

◦	8 dan 2 (test005_0)
◦	G dan C (test097_0: B1650NOG → B1650NOC)

◦	0 dan O (banyak kasus)
◦	B dan 8 (test004_0: raw output 89062VEH → dikoreksi ke B9062VEH, tetapi di beberapa kasus tetap salah)
6.	Delesi karakter akhir – model sering menghilangkan huruf terakhir jika berbentuk O, Q, atau G, misal E1517DQ → E1517.

7.	Kesalahan total – terjadi pada gambar dengan kualitas buruk, bounding box kurang tepat, atau plat dengan font tidak umum. Model terkadang menghasilkan pola yang sama sekali berbeda (misal L3823IB → O234XYZ).

4.5	Faktor Penyebab Kesalahan

8.	Ukuran model (2.2B) – masih terbatas dalam membedakan detail halus.

9.	Kualitas gambar – beberapa plat buram, terlalu terang/gelap, atau miring.

10.	Variasi font – plat dengan font dekoratif menyulitkan model.

11.	Teks tambahan – meskipun sudah diinstruksikan, model kadang masih menyertakan angka tanggal berlaku (misal AE8137G0624 tetapi berhasil di-post-process).

