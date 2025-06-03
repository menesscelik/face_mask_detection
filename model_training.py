# USAGE
# python train_mask_detector.py --dataset dataset

# Gerekli kütüphaneler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import os

# Argümanları al
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector_model.h5", help="path to output model file (.h5)")
args = vars(ap.parse_args())

# Eğitim parametreleri
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# Görüntüleri yükle
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Görüntüleri ve etiketleri al
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

# NumPy dizilerine çevir
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Etiketleri sayısal hale getir
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Eğitim/test ayrımı
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Veri artırma
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# MobileNetV2 base modeli
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Yeni baş katman
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Tam model
model = Model(inputs=baseModel.input, outputs=headModel)

# Base model donsun
for layer in baseModel.layers:
    layer.trainable = False

# Derleme
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Eğitim
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Değerlendirme
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Sınıflandırma raporu
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Modeli kaydet
print(f"[INFO] saving mask detector model as {args['model']}...")
model.save(args["model"])

# Grafik oluştur (zoomlu)
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(12, 6))

train_loss = H.history["loss"]
val_loss = H.history["val_loss"]
train_acc = H.history["accuracy"]
val_acc = H.history["val_accuracy"]
epochs = np.arange(1, N + 1)

plt.plot(epochs, train_loss, label="Train Loss", linewidth=2.0, marker='o', markersize=6)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2.0, marker='o', markersize=6)
plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2.0, marker='s', markersize=6)
plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2.0, marker='s', markersize=6)

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.01))

min_val = min(min(train_loss), min(val_loss), min(train_acc), min(val_acc))
max_val = max(max(train_loss), max(val_loss), max(train_acc), max(val_acc))
plt.ylim([min_val - 0.02, max_val + 0.02])

plt.title("Zoomed Training Loss and Accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss / Accuracy", fontsize=12)
plt.xticks(epochs)
plt.legend(loc="lower center", ncol=2)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(args["plot"])
plt.close()
print(f"[INFO] Grafik kaydedildi: {args['plot']}")
