
# 🧠 Yüz Maskesi Tespiti

Bu proje, derin öğrenme ve bilgisayarla görme tekniklerini kullanarak, görüntülerdeki kişilerin maske takıp takmadığını tespit etmeyi amaçlamaktadır.
Model, Keras ve TensorFlow kullanılarak eğitilmiş olup, gerçek zamanlı uygulamalar için uygundur.

---

## 📁 Proje Yapısı

Proje aşağıdaki dosya ve klasörleri içermektedir:

- `dataset/` – Eğitim ve test verilerini içeren klasör.
- `face_detector/` – Yüz tespiti için kullanılan önceden eğitilmiş modeller.
- `model_training.py` – Maske tespiti modelini eğitmek için kullanılan Python betiği.
- `mask_detector_model.h5` – Eğitilmiş modelin ağırlıklarını içeren dosya.
- `gui.py` – Gerçek zamanlı maske tespiti için grafiksel kullanıcı arayüzü.

---

## ⚙️ Kurulum

1. **Depoyu Klonlayın:**

```bash
git clone https://github.com/menesscelik/face_mask_detection.git
cd face_mask_detection
```

2. **Gerekli Kütüphaneleri Yükleyin:**

```bash
pip install -r requirements.txt
```

Eğer `requirements.txt` yoksa, şunları yükleyin:

```bash
pip install tensorflow keras opencv-python
```

3. **Modeli Eğitin:**

```bash
python model_training.py
```

4. **Uygulamayı Başlatın:**

```bash
python gui.py
```

---

## 🧪 Test

`gui.py` çalıştırıldığında, web kameranızı kullanarak yüzleri tespit edecek ve maske takılıp takılmadığını belirleyecektir.

---

## 📌 Notlar

- `face_detector/` klasörü gerekli yüz tespit modellerini içermelidir.
- `dataset/` klasörü maske takan ve takmayan kişilerin görüntülerini içermelidir.

---

## 📄 Lisans

Bu proje, MIT Lisansı altında lisanslanmıştır.
