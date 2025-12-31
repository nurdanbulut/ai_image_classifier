# 🧠 Animals-10 Görüntü Sınıflandırma Projesi

Bu proje, **Animals-10** veri seti kullanılarak 10 farklı hayvan sınıfının
görüntüler üzerinden sınıflandırılmasını amaçlamaktadır.

Model, **MobileNetV2** mimarisi ve **Transfer Learning** yaklaşımı kullanılarak
PyTorch framework’ü ile eğitilmiş; eğitim sonrası model performansı
çeşitli metrikler ile değerlendirilmiştir.
Ayrıca, kullanıcıların modeli etkileşimli şekilde test edebilmesi için
**Streamlit tabanlı modern bir web arayüzü** geliştirilmiştir.
<img width="1065" height="816" alt="Ekran görüntüsü 2025-12-30 194146" src="https://github.com/user-attachments/assets/17e82831-ff98-439f-b9c4-b3fe5eb72636" />



---

## 🎯 Proje Amaçları

- Görüntü sınıflandırma problemi için uçtan uca bir yapay zeka pipeline’ı kurmak
- Transfer Learning yaklaşımını pratikte uygulamak
- Eğitim, doğrulama ve test süreçlerini net şekilde ayırmak
- Model performansını metrikler ve görseller ile analiz etmek
- Modeli kullanıcı dostu bir arayüz üzerinden sunmak

---

## 🐾 Veri Seti

- **Adı:** Animals-10
- **Sınıf Sayısı:** 10  
  (`cane`, `cavallo`, `elefante`, `farfalla`, `gallina`, `gatto`, `mucca`, `pecora`, `ragno`, `scoiattolo`)
- **Görsel Türü:** RGB görüntüler
- **Kaynak:**  
  https://www.kaggle.com/datasets/alessiocorrado99/animals10

> ⚠️ Veri seti GitHub reposuna dahil edilmemiştir.


---

## 🔄 Veri Ön İşleme ve Ayrım (Dataset Split)

Veri seti, modelin genelleme yeteneğini ölçebilmek amacıyla üç parçaya ayrılmıştır:

- **Training set:** %70
- **Validation set:** %15
- **Test set:** %15

Bu işlem `src/dataset_split.py` dosyası ile otomatik olarak gerçekleştirilmiştir.
Ayrım sonrası her sınıf için örnek sayıları raporlanmıştır.

---

## 🧠 Model Mimarisi

- **Model:** MobileNetV2
- **Yaklaşım:** Transfer Learning
- **Ön Eğitim:** ImageNet
- **Son Katman:** 10 sınıflı Fully Connected katman

MobileNetV2, düşük parametre sayısı ve yüksek verimliliği sayesinde
CPU ortamında çalışmaya uygun bir mimaridir.

---

## ⚙️ Eğitim Süreci

- **Framework:** PyTorch
- **Epoch:** 10
- **Batch Size:** 16
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Çalışma Modu:** CPU

Eğitim sırasında:
- Train accuracy & loss
- Validation accuracy & loss
takip edilmiştir.

Eğitim sürecine ait grafikler:


<img width="640" height="480" alt="accuracy_curve" src="https://github.com/user-attachments/assets/3ca14ab2-c68d-42e5-8184-c0f03f673e59" />

<img width="640" height="480" alt="loss_curve" src="https://github.com/user-attachments/assets/82dc3064-d28b-4ede-8ee4-2525fe982e1e" />

---

## 📊 Değerlendirme ve Metrikler

Model, **test veri seti** üzerinde değerlendirilmiştir.

Kullanılan metrikler:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

<img width="700" height="427" alt="image" src="https://github.com/user-attachments/assets/ddeb0561-5a67-43b1-b2ff-a3e7620c0cf6" />

Özet sonuçlar:
- **Test Accuracy:** %95.7
- **Macro F1-score:** %95.1
- **Weighted F1-score:** %95.7


---

## 🖥️ Web Arayüzü (Streamlit)

Model, Streamlit kullanılarak geliştirilen bir web arayüzü ile sunulmuştur.

Arayüz özellikleri:
- Görsel yükleme (JPG / PNG)
- Tahmin edilen sınıfın Türkçe gösterimi
- Güven skoru (confidence)
- Top-3 sınıf olasılıkları

<img width="1080" height="955" alt="Ekran görüntüsü 2025-12-30 194127" src="https://github.com/user-attachments/assets/7f3d2d3a-a6d9-444c-a6c5-706d64f0d3e9" />

## 🎥 Uygulama Demo Videosu

📽️ Uygulamanın çalışma anına ait demo videosunu izlemek için aşağıdaki bağlantıya tıklayabilirsiniz:

👉 [Demo videosunu izlemek için tıklayın](https://github.com/nurdanbulut/Animal_image_classifier/blob/main/app/assets/20251231_125335.mp4)


---

## 🚀 Kurulum ve Çalıştırma

### 1️⃣ Sanal ortam oluşturma
```bash
python -m venv .venv
```

### 2️⃣ Sanal ortamı aktif etme (Windows)
```
.\.venv\Scripts\activate
```

### 3️⃣ Gerekli kütüphaneleri kurma
```
pip install -r requirements.txt
```

### 4️⃣ Uygulamayı çalıştırma
```
python -m streamlit run app/app.py
```
Uygulama şu adreste açılır:
http://localhost:8501


## 📁 Proje Yapısı
```
ai_image_classifier/
├── app/
│   ├── app.py
│   ├── style.css
│   └── assets/
│       └── hero.png
├── src/
│   ├── dataset_split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── plot_logs.py
│   └── config.py
├── models/
│   └── best_model.pth
├── outputs/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── README.md
```
