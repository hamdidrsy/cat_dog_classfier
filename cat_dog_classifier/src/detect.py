import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === AYARLAR ===
model_path = "models/cat_dog_model.h5"
test_dir = "data/test1"
img_size = 128

# === MODELİ YÜKLE ===
print("📦 Model yükleniyor...")
model = load_model(model_path)
print("✅ Model yüklendi.")

# === TAHMİN ===
def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"HATA: Görsel bulunamadı → {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"HATA: Görsel okunamadı → {img_path}")
        return

    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img / 255.0, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]
    label = "köpek" if prediction > 0.5 else "kedi"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"📷 {os.path.basename(img_path)} → Tahmin: {label} - Güven: {confidence:.2f}")

# === ÖRNEK KULLANIM ===
predict_image(os.path.join(test_dir, "5940.jpg"))
predict_image(os.path.join(test_dir, "6020.jpg"))
