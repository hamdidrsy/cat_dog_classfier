import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === AYARLAR ===
data_dir = "data/train"
img_size = 128
epochs = 15

# === VERİYİ YÜKLE ===
X, y = [], []
print("🔄 Veri yükleniyor...")

for filename in os.listdir(data_dir):
    if filename.endswith(".jpg"):
        path = os.path.join(data_dir, filename)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size, img_size))
        label = 0 if "cat" in filename.lower() else 1
        X.append(img)
        y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

print(f"✅ Toplam resim sayısı: {len(X)}")

# === VERİYİ AYIR ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL OLUŞTUR ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === EARLY STOPPING ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === EĞİT ===
print("\n🚀 Eğitim başlatıldı...")
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# === KAYDET ===
os.makedirs("models", exist_ok=True)
model.save("models/cat_dog_model.h5")
print("✅ Model 'models/' klasörüne kaydedildi.")

# === GRAFİK ===
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title("Model Doğruluğu")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
