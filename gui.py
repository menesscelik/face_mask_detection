import sys
import os
import cv2
import pickle
import time
import csv
import numpy as np
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

USERS_DIR = "users"
MASKLESS_PHOTO_DIR = os.path.join(USERS_DIR, "maskless_photos")
os.makedirs(USERS_DIR, exist_ok=True)
os.makedirs(MASKLESS_PHOTO_DIR, exist_ok=True)

# OpenCV DNN face detector
prototxt_path = os.path.join("face_detector", "deploy.prototxt")
weights_path = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

class MaskDetectionApp(QWidget):
    def __init__(self, camera_index=1):
        super().__init__()
        self.setWindowTitle("Face Mask Detection")
        self.setGeometry(100, 100, 800, 600)

        self.camera_index = camera_index
        self.capture = None

        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)

        self.name_input = QLineEdit(self)
        self.name_input.setAlignment(Qt.AlignCenter)
        self.name_input.setPlaceholderText("Enter Username")

        self.register_button = QPushButton("sign up")
        self.login_button = QPushButton("sign in")
        self.start_register_button = QPushButton("Start Recording")
        self.logout_button = QPushButton("logout")
        self.back_to_menu_button = QPushButton("Back to Main Menu")

        self.register_button.clicked.connect(self.start_camera_and_register)
        self.login_button.clicked.connect(self.start_camera_and_login)
        self.logout_button.clicked.connect(self.logout_user)
        self.start_register_button.clicked.connect(self.register_user)
        self.back_to_menu_button.clicked.connect(self.return_to_main_menu)

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.register_button)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.name_input)
        self.layout.addWidget(self.start_register_button)
        self.layout.addWidget(self.logout_button)
        self.layout.addWidget(self.back_to_menu_button)
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.current_frame = None
        self.logged_in = False
        self.logged_in_user = None

        self.mask_model = load_model("mask_detector_model.h5")
        self.facenet = FaceNet()
        self.known_users = []
        self.last_check_time = 0
        self.mask_status_cache = {}
        self.maskless_log_cache = {}
        self.maskless_photo_dir = MASKLESS_PHOTO_DIR

        self.reset_to_main_menu()

    def reset_to_main_menu(self):
        self.clear_interface()
        self.register_button.show()
        self.login_button.setText("Sign in")
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.start_camera_and_login)
        self.login_button.show()

    def clear_interface(self):
        self.name_input.hide()
        self.name_input.clear()
        self.login_button.hide()
        self.logout_button.hide()
        self.start_register_button.hide()
        self.back_to_menu_button.hide()
        self.register_button.hide()

    def return_to_main_menu(self):
        self.release_camera()
        self.reset_to_main_menu()

    def init_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(self.camera_index)
            self.timer.start(30)

    def release_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.timer.stop()
            self.video_label.clear()

    def start_camera_and_login(self):
        self.init_camera()
        self.clear_interface()
        self.name_input.hide()
        self.login_button.setText("Sign in")
        self.login_button.show()
        self.login_button.clicked.disconnect()
        self.login_button.clicked.connect(self.login_user)

    def start_camera_and_register(self):
        self.init_camera()
        self.clear_interface()
        self.name_input.show()
        self.name_input.setFocus()
        self.start_register_button.show()

    def detect_faces_dnn(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes.append(box.astype("int"))
        return boxes

    def detect_single_face(self, rgb_frame):
        boxes = self.detect_faces_dnn(rgb_frame)
        if len(boxes) != 1:
            return None
        (startX, startY, endX, endY) = boxes[0]
        face = rgb_frame[startY:endY, startX:endX]
        face = cv2.resize(face, (160, 160))
        return face

    def register_user(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir kullanıcı adı girin.")
            return

        if self.current_frame is not None:
            QMessageBox.information(self, "Bilgi", "1. Aşama: Maskesiz yüzünüzle kayıt olun.")
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            boxes = self.detect_faces_dnn(rgb_frame)
            if len(boxes) != 1:
                QMessageBox.warning(self, "Uyarı", "Lütfen yalnızca bir yüzle kaydolun.")
                return
            (startX, startY, endX, endY) = boxes[0]
            face = rgb_frame[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            emb_nomask = self.facenet.embeddings([face])[0]

            QMessageBox.information(self, "Bilgi", "2. Aşama: Maskeli halinizle tekrar bakın ve 'Tamam' deyin.")
            QMessageBox.information(self, "Hazır Mısınız?", "Maskenizi taktıktan sonra OK tuşuna basın.")

            ret, frame = self.capture.read()
            if not ret:
                QMessageBox.warning(self, "Hata", "Kameradan görüntü alınamadı.")
                return
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = self.detect_faces_dnn(rgb_frame)
            if len(boxes) != 1:
                QMessageBox.warning(self, "Uyarı", "Maskeli halde yalnızca bir yüz olmalı.")
                return
            (startX, startY, endX, endY) = boxes[0]
            face = rgb_frame[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            emb_mask = self.facenet.embeddings([face])[0]

            user_data = {"name": name, "embeddings": [emb_nomask, emb_mask]}
            with open(os.path.join(USERS_DIR, f"{name}.pkl"), "wb") as f:
                pickle.dump(user_data, f)

            QMessageBox.information(self, "Başarılı", f"{name} için veriler kaydedildi.")
            self.return_to_main_menu()

    def login_user(self):
        self.known_users.clear()
        for filename in os.listdir(USERS_DIR):
            if filename.endswith(".pkl"):
                with open(os.path.join(USERS_DIR, filename), "rb") as f:
                    self.known_users.append(pickle.load(f))

        if not self.known_users:
            QMessageBox.warning(self, "Hata", "Hiç kayıtlı kullanıcı yok.")
            return

        if self.current_frame is not None:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            face = self.detect_single_face(rgb_frame)
            if face is None:
                QMessageBox.warning(self, "Hata", "Yüz algılanamadı.")
                return
            input_emb = self.facenet.embeddings([face])[0]

            best_match = None
            best_distance = 1.0
            for user in self.known_users:
                for emb in user["embeddings"]:
                    dist = np.linalg.norm(input_emb - emb)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = user["name"]

            if best_distance < 0.8:
                QMessageBox.information(self, "Başarılı", f"{best_match} olarak giriş yapıldı.")
                self.clear_interface()
                self.back_to_menu_button.show()
            else:
                QMessageBox.warning(self, "Hata", "Yüz eşleşmedi.")

    def log_maskless_user(self, username):
        log_path = os.path.join(USERS_DIR, "maskless_log.csv")
        current_time = time.time()

        last_logged = self.maskless_log_cache.get(username, 0)
        if current_time - last_logged < 180:
            return

        self.maskless_log_cache[username] = current_time
        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(current_time))
        timestamp_csv = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Isim", "Tarih/Saat"])
            writer.writerow([username, timestamp_csv])
            print(f"[LOG] {username} maskesiz olarak kaydedildi: {timestamp_csv}")

        if self.current_frame is not None and hasattr(self, 'last_boxes'):
            annotated = self.current_frame.copy()
            for (startX, startY, endX, endY) in self.last_boxes:
                cv2.rectangle(annotated, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(annotated, f"{username} - Maskesiz", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            photo_path = os.path.join(self.maskless_photo_dir, f"{username}_{timestamp_str}.jpg")
            cv2.imwrite(photo_path, annotated)
            print(f"[FOTO] Fotoğraf kaydedildi: {photo_path}")

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.current_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()

        run_check = current_time - self.last_check_time >= 0.3
        if run_check:
            self.last_check_time = current_time
            boxes = self.detect_faces_dnn(rgb_frame)
            self.last_boxes = boxes
        else:
            boxes = getattr(self, 'last_boxes', [])

        for (startX, startY, endX, endY) in boxes:
            face = rgb_frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            try:
                face_resized = cv2.resize(face, (160, 160))
            except:
                continue
            embedding = self.facenet.embeddings([face_resized])[0]

            label = "Bilinmiyor"
            color = (255, 255, 0)

            min_dist = 1.0
            best_match = None
            for user in self.known_users:
                for emb in user['embeddings']:
                    dist = np.linalg.norm(embedding - emb)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = user['name']

            if best_match and min_dist < 0.8:
                self.logged_in = True
                self.logged_in_user = best_match

                face_id = (startX, startY, endX, endY)
                if run_check:
                    try:
                        face_crop = frame[startY:endY, startX:endX]
                        resized = cv2.resize(face_crop, (224, 224))
                        face_array = resized.astype("float32") / 255.0
                        face_array = np.expand_dims(face_array, axis=0)
                        (mask, withoutMask) = self.mask_model.predict(face_array, verbose=0)[0]
                        mask_status = "Maskeli" if mask > withoutMask else "Maskesiz"
                        self.mask_status_cache[face_id] = (mask_status, (0, 255, 0) if mask > withoutMask else (0, 0, 255))
                    except:
                        continue

                if face_id in self.mask_status_cache:
                    mask_status, color = self.mask_status_cache[face_id]
                    label = f"{best_match}: {mask_status}"

                    if mask_status == "Maskesiz":
                        self.log_maskless_user(best_match)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        qt_img = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data,
                        frame.shape[1], frame.shape[0],
                        frame.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))

    def logout_user(self):
        self.logged_in = False
        self.logged_in_user = None
        self.return_to_main_menu()

    def closeEvent(self, event):
        self.release_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Modern font ve genel stil ayarları
    app.setStyleSheet('''
        QWidget {
            background-color: #f4f6fb;
            font-family: 'Segoe UI', 'Roboto', 'Arial', 'Helvetica Neue', sans-serif;
            font-size: 16px;
            color: #222;
        }
        QLabel {
            font-size: 18px;
            color: #222;
        }
        QLineEdit {
            border: 1.5px solid #bfc7d5;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 16px;
            background: #fff;
            color: #222;
        }
        QPushButton {
            background-color: #4f8cff;
            color: #fff;
            border: none;
            border-radius: 10px;
            padding: 12px 0;
            font-size: 17px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(79,140,255,0.08);
            transition: background 0.2s, box-shadow 0.2s, transform 0.1s;
        }
        QPushButton:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 16px rgba(37,99,235,0.15);
            transform: translateY(-2px) scale(1.03);
        }
        QPushButton:pressed {
            background-color: #174ea6;
            box-shadow: 0 1px 4px rgba(23,78,166,0.12);
            transform: scale(0.98);
        }
        QMessageBox QLabel {
            font-size: 16px;
        }
    ''')
    window = MaskDetectionApp(camera_index=1)
    window.show()
    sys.exit(app.exec_())
