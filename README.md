
# 🧠 Face Mask Detection

This project aims to detect whether individuals in images are wearing face masks using deep learning and computer vision techniques.
The model is trained using Keras and TensorFlow and is suitable for real-time applications.

---

## 📁 Project Structure

The project includes the following files and directories:

- `dataset/` – Contains training and test data.
- `face_detector/` – Pre-trained models used for face detection.
- `model_training.py` – Python script used to train the mask detection model.
- `mask_detector_model.h5` – File containing the trained model weights.
- `gui.py` – Graphical user interface for real-time mask detection.

---

## ⚙️ Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/menesscelik/face_mask_detection.git
cd face_mask_detection
```


If `requirements.txt` is missing, manually install the following:

```bash
pip install tensorflow keras opencv-python
```

3. **Train the Model:**

```bash
python model_training.py
```

4. **Run the Application:**

```bash
python gui.py
```

---

## 🧪 Testing

Once you run `gui.py`, it will use your webcam to detect faces and determine whether a mask is being worn.

---

## 📌 Notes

- The `face_detector/` folder should contain the necessary face detection models.
- The `dataset/` folder should contain images of people with and without masks.

---

## 📄 License

This project is licensed under the MIT License.
