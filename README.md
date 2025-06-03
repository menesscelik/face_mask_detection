# 🧠 Face Mask Detection System

This project is a comprehensive implementation of a **real-time face mask detection system** using deep learning and computer vision technologies. The system can identify whether a person is wearing a face mask based on live webcam input or static images, making it useful in public safety environments such as offices, airports, and hospitals.

---

## 🎯 Objective

- Build a classifier that distinguishes between masked and unmasked faces.
- Enable real-time mask detection using webcam video streams.
- Provide a modular and extensible codebase for training and inference.

---

## 🧩 Project Structure

```
face_mask_detection/
│
├── dataset/                  # Training and testing image datasets
│   ├── with_mask/
│   └── without_mask/
│
├── face_detector/            # Pre-trained face detection models
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── model_training.py         # Training script for mask detection model
├── mask_detector_model.h5    # Trained Keras model
├── gui.py                    # Real-time detection interface
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/menesscelik/face_mask_detection.git
cd face_mask_detection
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install tensorflow keras opencv-python imutils numpy matplotlib
```

---

## 🏋️‍♂️ Training the Model

```bash
python model_training.py
```

- Loads data from `dataset/`
- Preprocesses images and labels
- Builds and trains a CNN using Keras
- Saves the trained model as `mask_detector_model.h5`

---

## 🧪 Running the Application

```bash
python gui.py
```

- Starts webcam capture
- Detects faces and classifies as "Mask" or "No Mask"
- Displays colored bounding boxes around detected faces

---

## 💡 Customization

- Add more images to `dataset/` for better training
- Swap in different architectures (e.g., MobileNet, EfficientNet)
- Convert model to TensorFlow Lite for deployment on mobile devices

---

## 🛠 Troubleshooting

- Ensure `face_detector/` contains correct model files
- Check webcam access and permissions
- Confirm TensorFlow and Keras versions are compatible

---

## 📌 Notes

- `face_detector/` must contain:
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`
- If missing, download from OpenCV GitHub:
  https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project.

---

## 🙋‍♀️ Author

Developed by [@menesscelik](https://github.com/menesscelik)
