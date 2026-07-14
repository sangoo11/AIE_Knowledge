# 🧠 Dự Án: Nhận Dạng Chữ & Số Viết Tay (Handwriting Recognition)

> **Mục tiêu**: Xây dựng mô hình AI có thể dự đoán chữ cái (A-Z) hoặc chữ số (0-9) từ ảnh nền trắng, chữ đen.
> **Công cụ**: Google Colab (miễn phí GPU)
> **Thời gian ước tính**: 3-5 giờ cho người mới bắt đầu
> **Ngôn ngữ**: Python
> **Framework**: TensorFlow / Keras

---

## 📋 Tổng Quan Dự Án

```
Input:  Ảnh nền trắng, chữ/số đen (28x28 pixels, grayscale)
Model:  Convolutional Neural Network (CNN)
Output: Dự đoán ký tự đó là gì (0-9 hoặc A-Z)
```

### Kiến trúc tổng quan:

```
Ảnh đầu vào (28x28)
    ↓
[Conv2D → ReLU → MaxPool] × 2
    ↓
[Flatten]
    ↓
[Dense → ReLU → Dropout]
    ↓
[Dense → Softmax]
    ↓
Kết quả dự đoán
```

---

## 📚 Kiến Thức Cần Biết Trước

| Khái niệm | Giải thích đơn giản |
|---|---|
| **CNN** | Mạng nơ-ron chuyên xử lý ảnh, tự học các đặc trưng (cạnh, góc, hình dạng) |
| **Conv2D** | Lớp tích chập - quét ảnh bằng bộ lọc để tìm đặc trưng |
| **MaxPooling** | Giảm kích thước ảnh, giữ lại thông tin quan trọng nhất |
| **Flatten** | Chuyển ma trận 2D thành vector 1D để đưa vào Dense |
| **Dense** | Lớp fully-connected, mỗi neuron kết nối với tất cả neuron trước đó |
| **Dropout** | Tắt ngẫu nhiên một số neuron khi train → chống overfitting |
| **Softmax** | Chuyển output thành xác suất (tổng = 1) |
| **EMNIST** | Bộ dataset chứa ảnh chữ viết tay (chữ + số), mở rộng từ MNIST |

---

## 🗂️ Cấu Trúc Các Bước

```
Phase 1: Thiết lập môi trường           (Bước 1-2)
Phase 2: Chuẩn bị dữ liệu              (Bước 3-5)
Phase 3: Xây dựng mô hình              (Bước 6-7)
Phase 4: Huấn luyện mô hình            (Bước 8-9)
Phase 5: Đánh giá & Phân tích          (Bước 10-12)
Phase 6: Dự đoán ảnh thực tế           (Bước 13-15)
Phase 7: Lưu & Triển khai              (Bước 16-17)
```

---

---

# PHASE 1: THIẾT LẬP MÔI TRƯỜNG

---

## Bước 1: Tạo Google Colab Notebook

### 1.1 Mở Google Colab
1. Truy cập: [https://colab.research.google.com](https://colab.research.google.com)
2. Đăng nhập bằng tài khoản Google
3. Click **"New Notebook"** (Notebook mới)
4. Đổi tên notebook: Click vào tên ở góc trên trái → đặt tên `Handwriting_Recognition.ipynb`

### 1.2 Bật GPU (Quan trọng - giúp train nhanh hơn ~10x)
1. Vào menu **Runtime** → **Change runtime type**
2. Chọn **Hardware accelerator** → **T4 GPU**
3. Click **Save**

### 1.3 Kiểm tra GPU đã bật chưa

Tạo cell đầu tiên và chạy:

```python
# ============================================================
# CELL 1: Kiểm tra GPU
# ============================================================
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("✅ GPU đã được kích hoạt! Training sẽ nhanh hơn.")
else:
    print("⚠️ Không có GPU. Vẫn chạy được nhưng sẽ chậm hơn.")
```

**✅ Output mong đợi:**
```
TensorFlow version: 2.x.x
GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
✅ GPU đã được kích hoạt! Training sẽ nhanh hơn.
```

---

## Bước 2: Import Thư Viện

Tạo cell mới và chạy:

```python
# ============================================================
# CELL 2: Import tất cả thư viện cần thiết
# ============================================================

# --- Thư viện chính ---
import numpy as np                  # Xử lý mảng số
import matplotlib.pyplot as plt     # Vẽ biểu đồ, hiển thị ảnh
import tensorflow as tf             # Framework deep learning
from tensorflow import keras        # API cao cấp của TensorFlow
from tensorflow.keras import layers # Các lớp neural network

# --- Thư viện xử lý ảnh ---
from PIL import Image               # Đọc và xử lý ảnh
import cv2                          # OpenCV - xử lý ảnh nâng cao

# --- Thư viện đánh giá ---
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns               # Vẽ biểu đồ đẹp hơn

# --- Thư viện tiện ích ---
import os
import warnings
warnings.filterwarnings('ignore')   # Ẩn warning không cần thiết

# --- Cài thêm thư viện EMNIST ---
!pip install emnist -q              # -q = quiet, ít output hơn
from emnist import extract_training_samples, extract_test_samples

print("✅ Tất cả thư viện đã được import thành công!")
```

**✅ Output mong đợi:**
```
✅ Tất cả thư viện đã được import thành công!
```

> **💡 Giải thích**: Nếu gặp lỗi `ModuleNotFoundError`, chạy `!pip install <tên_thư_viện>` trong 1 cell riêng rồi chạy lại cell import.

---

---

# PHASE 2: CHUẨN BỊ DỮ LIỆU

---

## Bước 3: Tải Dataset EMNIST

> **Tại sao dùng EMNIST?**
> - MNIST chỉ có số (0-9) = 10 classes
> - EMNIST Balanced có cả chữ lẫn số = 47 classes
> - Dataset đã được chuẩn hóa (28x28 pixels, grayscale, nền đen chữ trắng)
> - Rất phổ biến, dễ sử dụng cho người mới

```python
# ============================================================
# CELL 3: Tải dữ liệu EMNIST (Balanced dataset)
# ============================================================
# Do link tải mặc định của thư viện có thể bị lỗi (chặn truy cập),
# ta sẽ tải thủ công file zip chứa dữ liệu gốc trước.

import os

# 1. Tạo thư mục cache và xóa file lỗi (nếu có)
!mkdir -p ~/.cache/emnist
!rm -f ~/.cache/emnist/emnist.zip

# 2. Tải thủ công file dataset gzip (~536MB)
print("Đang tải dữ liệu gốc (gzip.zip)...")
!wget --user-agent="Mozilla/5.0" -q --show-progress \
      -O ~/.cache/emnist/emnist.zip \
      "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

print("\nĐã tải xong! Đang trích xuất dữ liệu...")

# 3. Tải dữ liệu training & test (thư viện sẽ dùng file vừa tải)
X_train, y_train = extract_training_samples('balanced')
X_test, y_test = extract_test_samples('balanced')

print("=" * 50)
print("📊 THÔNG TIN DATASET")
print("=" * 50)
print(f"Training set:  {X_train.shape[0]:,} ảnh")
print(f"Test set:      {X_test.shape[0]:,} ảnh")
print(f"Kích thước ảnh: {X_train.shape[1]}x{X_train.shape[2]} pixels")
print(f"Số classes:    {len(np.unique(y_train))}")
print(f"Giá trị pixel: min={X_train.min()}, max={X_train.max()}")
```

**✅ Output mong đợi:**
```
==================================================
📊 THÔNG TIN DATASET
==================================================
Training set:  112,800 ảnh
Test set:      18,800 ảnh
Kích thước ảnh: 28x28 pixels
Số classes:    47
Giá trị pixel: min=0, max=255
```

---

## Bước 4: Tìm Hiểu Các Class (Label Mapping)

```python
# ============================================================
# CELL 4: Tạo mapping từ label (số) → ký tự thực tế
# ============================================================

# EMNIST Balanced có 47 classes:
# 0-9:   chữ số 0-9
# 10-35: chữ cái A-Z (viết hoa)
# 36-46: một số chữ cái viết thường dễ nhầm lẫn được gộp

# Mapping chính thức của EMNIST Balanced
emnist_labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

# Tạo dictionary để tra cứu nhanh
label_to_char = {i: char for i, char in enumerate(emnist_labels)}
char_to_label = {char: i for i, char in enumerate(emnist_labels)}

num_classes = len(emnist_labels)

print(f"Tổng số classes: {num_classes}")
print(f"\nDanh sách classes:")
for i, char in enumerate(emnist_labels):
    print(f"  Label {i:2d} → '{char}'", end="    ")
    if (i + 1) % 6 == 0:
        print()  # Xuống dòng mỗi 6 items

print(f"\n\n💡 Ví dụ: Label 0 = '{label_to_char[0]}', Label 10 = '{label_to_char[10]}', Label 36 = '{label_to_char[36]}'")
```

**✅ Output mong đợi:**
```
Tổng số classes: 47

Danh sách classes:
  Label  0 → '0'    Label  1 → '1'    Label  2 → '2'    Label  3 → '3'    Label  4 → '4'    Label  5 → '5'
  Label  6 → '6'    Label  7 → '7'    Label  8 → '8'    Label  9 → '9'    Label 10 → 'A'    Label 11 → 'B'
  ...
```

---

## Bước 5: Trực Quan Hóa Dữ Liệu (Xem Thử Ảnh)

```python
# ============================================================
# CELL 5: Hiển thị mẫu ảnh từ dataset
# ============================================================

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("Mẫu ảnh từ EMNIST Balanced Dataset", fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    # Chọn ngẫu nhiên 1 ảnh
    idx = np.random.randint(0, len(X_train))
    
    # Hiển thị ảnh (cmap='gray' vì ảnh grayscale)
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f"'{label_to_char[y_train[idx]]}'", fontsize=10, color='blue')
    ax.axis('off')  # Ẩn trục tọa độ

plt.tight_layout()
plt.show()

print("💡 Lưu ý: Ảnh gốc EMNIST có nền ĐEN, chữ TRẮNG.")
print("   Ta sẽ đảo ngược (invert) khi dự đoán ảnh nền trắng chữ đen.")
```

**✅ Output mong đợi:** Hiển thị lưới 3×10 ảnh chữ/số viết tay với nhãn tương ứng.

---

```python
# ============================================================
# CELL 6: Tiền xử lý dữ liệu (Data Preprocessing)
# ============================================================

# --- Bước 5a: Chuẩn hóa pixel từ [0, 255] → [0, 1] ---
# Tại sao? Neural network hoạt động tốt hơn với giá trị nhỏ
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0

# --- Bước 5b: Thêm chiều channel ---
# CNN cần input shape (batch, height, width, channels)
# Ảnh grayscale có 1 channel
X_train = X_train.reshape(-1, 28, 28, 1)  # -1 = tự tính số ảnh
X_test  = X_test.reshape(-1, 28, 28, 1)

# --- Bước 5c: One-Hot Encoding cho labels ---
# Chuyển label từ số nguyên → vector nhị phân
# Ví dụ: label 3 → [0, 0, 0, 1, 0, 0, ..., 0]
y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
y_test_encoded  = keras.utils.to_categorical(y_test, num_classes)

print("=" * 50)
print("📊 SAU KHI TIỀN XỬ LÝ")
print("=" * 50)
print(f"X_train shape: {X_train.shape}  → (số ảnh, height, width, channels)")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train_encoded.shape}  → (số ảnh, num_classes)")
print(f"y_test shape:  {y_test_encoded.shape}")
print(f"Giá trị pixel: min={X_train.min():.1f}, max={X_train.max():.1f}")
print(f"\nVí dụ one-hot encoding:")
print(f"  Label gốc: {y_train[0]} ('{label_to_char[y_train[0]]}')")
print(f"  One-hot:   {y_train_encoded[0]}")
```

**✅ Output mong đợi:**
```
==================================================
📊 SAU KHI TIỀN XỬ LÝ
==================================================
X_train shape: (112800, 28, 28, 1)  → (số ảnh, height, width, channels)
X_test shape:  (18800, 28, 28, 1)
y_train shape: (112800, 47)  → (số ảnh, num_classes)
y_test shape:  (18800, 47)
Giá trị pixel: min=0.0, max=1.0
```

---

---

# PHASE 3: XÂY DỰNG MÔ HÌNH

---

## Bước 6: Thiết Kế Kiến Trúc CNN

```python
# ============================================================
# CELL 7: Xây dựng mô hình CNN
# ============================================================

def build_model(num_classes):
    """
    Xây dựng CNN model cho nhận dạng ký tự viết tay.
    
    Kiến trúc:
    - 2 khối Conv2D + MaxPooling (trích xuất đặc trưng)
    - 1 khối Dense + Dropout (phân loại)
    - Output layer với Softmax (xác suất)
    """
    model = keras.Sequential([
        # ===== INPUT LAYER =====
        keras.Input(shape=(28, 28, 1)),  # Ảnh 28x28, 1 channel (grayscale)
        
        # ===== KHỐI 1: Trích xuất đặc trưng cơ bản =====
        # Conv2D: 32 bộ lọc 3x3, tìm các đặc trưng đơn giản (cạnh, góc)
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # BatchNormalization: Chuẩn hóa output, giúp train ổn định hơn
        layers.BatchNormalization(),
        # Conv2D thứ 2: Tìm thêm đặc trưng
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # MaxPooling: Giảm kích thước 28x28 → 14x14
        layers.MaxPooling2D((2, 2)),
        # Dropout: Tắt 25% neuron ngẫu nhiên → chống overfitting
        layers.Dropout(0.25),
        
        # ===== KHỐI 2: Trích xuất đặc trưng phức tạp =====
        # Conv2D: 64 bộ lọc, tìm đặc trưng phức tạp hơn (hình dạng, nét chữ)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # MaxPooling: Giảm kích thước 14x14 → 7x7
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ===== KHỐI 3: Trích xuất đặc trưng cao cấp =====
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # MaxPooling: Giảm kích thước 7x7 → 3x3
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ===== CLASSIFICATION LAYERS =====
        # Flatten: Chuyển 3D (3x3x128) → 1D (1152)
        layers.Flatten(),
        
        # Dense: Fully connected layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),  # Dropout cao hơn ở FC layer
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # ===== OUTPUT LAYER =====
        # Softmax: Output xác suất cho mỗi class
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# --- Tạo model ---
model = build_model(num_classes)

# --- Xem tóm tắt kiến trúc ---
model.summary()
```

**✅ Output mong đợi:** Bảng tóm tắt model với tổng ~500K parameters.

---

## Bước 7: Compile Model (Cấu hình Training)

```python
# ============================================================
# CELL 8: Compile model - Cấu hình cách model học
# ============================================================

model.compile(
    # Optimizer: Adam - tự điều chỉnh learning rate
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    
    # Loss function: Categorical Crossentropy
    # Dùng cho bài toán phân loại nhiều class với one-hot encoding
    loss='categorical_crossentropy',
    
    # Metrics: Theo dõi accuracy trong quá trình training
    metrics=['accuracy']
)

print("✅ Model đã được compile!")
print(f"   Optimizer:      Adam (lr=0.001)")
print(f"   Loss function:  Categorical Crossentropy")
print(f"   Metrics:        Accuracy")
```

**✅ Output mong đợi:**
```
✅ Model đã được compile!
   Optimizer:      Adam (lr=0.001)
   Loss function:  Categorical Crossentropy
   Metrics:        Accuracy
```

---

---

# PHASE 4: HUẤN LUYỆN MÔ HÌNH

---

## Bước 8: Cấu Hình Callbacks (Chiến lược Training)

```python
# ============================================================
# CELL 9: Thiết lập callbacks
# ============================================================

callbacks = [
    # --- EarlyStopping ---
    # Dừng training sớm nếu model không cải thiện
    # patience=5: Chờ 5 epoch, nếu val_loss không giảm → dừng
    # restore_best_weights: Quay lại weights tốt nhất
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # --- ReduceLROnPlateau ---
    # Giảm learning rate khi model "bí" (val_loss không giảm)
    # factor=0.5: Giảm lr còn 1 nửa
    # patience=3: Chờ 3 epoch trước khi giảm
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
]

print("✅ Callbacks đã được cấu hình!")
print("   - EarlyStopping: Dừng nếu 5 epoch không cải thiện")
print("   - ReduceLROnPlateau: Giảm learning rate khi bí")
```

---

## Bước 9: Bắt Đầu Training

```python
# ============================================================
# CELL 10: TRAINING MODEL
# ============================================================

print("🚀 Bắt đầu training...\n")

history = model.fit(
    X_train, y_train_encoded,     # Dữ liệu training
    
    # --- Hyperparameters ---
    epochs=30,                     # Tối đa 30 epoch (EarlyStopping có thể dừng sớm)
    batch_size=128,                # Mỗi lần update weights, dùng 128 ảnh
    
    # --- Validation ---
    validation_split=0.15,         # Dùng 15% training data làm validation
    
    # --- Callbacks ---
    callbacks=callbacks,
    
    # --- Display ---
    verbose=1                      # Hiển thị progress bar
)

print("\n" + "=" * 50)
print("✅ TRAINING HOÀN TẤT!")
print("=" * 50)
print(f"Số epoch đã train: {len(history.history['loss'])}")
print(f"Train accuracy cuối: {history.history['accuracy'][-1]:.4f}")
print(f"Val accuracy cuối:   {history.history['val_accuracy'][-1]:.4f}")
```

**✅ Output mong đợi:**
```
🚀 Bắt đầu training...

Epoch 1/30
749/749 [==============================] - 15s - loss: 1.2xxx - accuracy: 0.6xxx - val_loss: 0.5xxx - val_accuracy: 0.8xxx
Epoch 2/30
...
(Training sẽ tự dừng khi EarlyStopping kích hoạt)

==================================================
✅ TRAINING HOÀN TẤT!
==================================================
Số epoch đã train: ~15-20
Train accuracy cuối: ~0.89
Val accuracy cuối:   ~0.86
```

> **⏱️ Thời gian**: ~5-10 phút với GPU trên Colab

---

---

# PHASE 5: ĐÁNH GIÁ & PHÂN TÍCH

---

## Bước 10: Vẽ Biểu Đồ Training (Loss & Accuracy)

```python
# ============================================================
# CELL 11: Vẽ biểu đồ training history
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Biểu đồ Accuracy ---
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# --- Biểu đồ Loss ---
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("💡 CÁCH ĐỌC BIỂU ĐỒ:")
print("   - Train & Val accuracy gần nhau → Model học tốt, không overfitting")
print("   - Val accuracy thấp hơn nhiều → Overfitting (model học thuộc)")
print("   - Cả 2 đều thấp → Underfitting (model chưa học đủ)")
```

**✅ Output mong đợi:** 2 biểu đồ cạnh nhau thể hiện accuracy tăng dần và loss giảm dần.

---

## Bước 11: Đánh Giá Trên Test Set

```python
# ============================================================
# CELL 12: Đánh giá model trên test set
# ============================================================

# --- Tính loss và accuracy trên test set ---
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

print("=" * 50)
print("📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET")
print("=" * 50)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# --- Dự đoán toàn bộ test set ---
y_pred_probs = model.predict(X_test, verbose=0)  # Xác suất
y_pred = np.argmax(y_pred_probs, axis=1)          # Class có xác suất cao nhất

# --- Classification Report ---
print(f"\n📋 CLASSIFICATION REPORT (Top 10 classes):")
print("-" * 50)

# In report cho 10 class đầu tiên
target_names = [f"'{label_to_char[i]}'" for i in range(num_classes)]
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

# Hiển thị tóm tắt
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 46)
for i in range(10):
    name = f"'{label_to_char[i]}'"
    p = report[name]['precision']
    r = report[name]['recall']
    f1 = report[name]['f1-score']
    print(f"{name:<10} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
print("...")
print(f"\n{'Overall Accuracy:':<25} {report['accuracy']:.4f}")
```

**✅ Output mong đợi:**
```
==================================================
📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET
==================================================
Test Loss:     ~0.4xxx
Test Accuracy: ~0.86xx (86.xx%)
```

---

## Bước 12: Confusion Matrix (Ma trận nhầm lẫn)

```python
# ============================================================
# CELL 13: Vẽ Confusion Matrix
# ============================================================

# Tính confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Vẽ heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    cm, 
    annot=False,           # Không hiển thị số (vì 47x47 quá nhiều)
    fmt='d', 
    cmap='Blues',
    xticklabels=emnist_labels,
    yticklabels=emnist_labels
)
plt.title('Confusion Matrix - EMNIST Balanced', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

# --- Tìm các cặp hay bị nhầm ---
print("\n🔍 TOP 10 CẶP HAY BỊ NHẦM LẪN:")
print("-" * 50)

# Tìm các cell có giá trị cao nhưng không nằm trên đường chéo
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)  # Bỏ đường chéo (đúng)

for _ in range(10):
    idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    true_char = emnist_labels[idx[0]]
    pred_char = emnist_labels[idx[1]]
    count = cm_copy[idx]
    print(f"  '{true_char}' bị nhầm thành '{pred_char}': {count} lần")
    cm_copy[idx] = 0
```

**✅ Output mong đợi:** Heatmap confusion matrix + danh sách các cặp ký tự hay bị nhầm (ví dụ: 'g' ↔ '9', 'l' ↔ '1').

---

---

# PHASE 6: DỰ ĐOÁN ẢNH THỰC TẾ

---

## Bước 13: Tạo Hàm Tiền Xử Lý Ảnh

> **⚠️ QUAN TRỌNG**: Đây là bước then chốt!
> - Dataset EMNIST: nền ĐEN, chữ TRẮNG
> - Ảnh thực tế của bạn: nền TRẮNG, chữ ĐEN
> - → Phải **đảo ngược** (invert) ảnh trước khi dự đoán
> - → EMNIST cũng bị **transpose** (xoay 90°), cần xử lý

```python
# ============================================================
# CELL 14: Hàm tiền xử lý ảnh đầu vào
# ============================================================

def preprocess_image(image_path):
    """
    Tiền xử lý ảnh từ file → format phù hợp với model.
    
    Pipeline:
    1. Đọc ảnh
    2. Chuyển sang grayscale
    3. Resize về 28x28
    4. Đảo ngược màu (trắng→đen, đen→trắng)
    5. Chuẩn hóa pixel [0, 1]
    6. Reshape cho model
    
    Args:
        image_path: Đường dẫn tới file ảnh
    
    Returns:
        processed_img: Ảnh đã xử lý, shape (1, 28, 28, 1)
        display_img:   Ảnh để hiển thị
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    
    # 2. Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize về 28x28 (kích thước dataset)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Lưu bản hiển thị
    display_img = resized.copy()
    
    # 4. ĐẢO NGƯỢC MÀU (Quan trọng!)
    # Ảnh gốc: nền trắng (255), chữ đen (0)
    # EMNIST:   nền đen (0),    chữ trắng (255)
    inverted = 255 - resized
    
    # 5. Chuẩn hóa [0, 255] → [0, 1]
    normalized = inverted.astype('float32') / 255.0
    
    # 6. Reshape: (28, 28) → (1, 28, 28, 1)
    # Thêm batch dimension và channel dimension
    processed = normalized.reshape(1, 28, 28, 1)
    
    return processed, display_img


def predict_character(image_path, model, label_map, top_k=5):
    """
    Dự đoán ký tự từ ảnh và hiển thị kết quả.
    
    Args:
        image_path: Đường dẫn ảnh
        model:      Model đã train
        label_map:  Dictionary label → character
        top_k:      Số lượng dự đoán top hiển thị
    """
    # Tiền xử lý
    processed_img, display_img = preprocess_image(image_path)
    
    # Dự đoán
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Top-K predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    
    # Hiển thị kết quả
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Ảnh gốc
    ax1.imshow(display_img, cmap='gray')
    ax1.set_title("Ảnh đầu vào", fontsize=14)
    ax1.axis('off')
    
    # Bar chart xác suất
    chars = [label_map[i] for i in top_indices]
    probs = [predictions[i] * 100 for i in top_indices]
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(top_k)]
    
    bars = ax2.barh(range(top_k), probs, color=colors)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([f"'{c}'" for c in chars], fontsize=14)
    ax2.set_xlabel('Xác suất (%)', fontsize=12)
    ax2.set_title('Top Dự Đoán', fontsize=14)
    ax2.invert_yaxis()
    
    # Thêm % lên bars
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob:.1f}%', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # In kết quả
    print(f"\n🎯 KẾT QUẢ: Ký tự dự đoán là '{label_map[top_indices[0]]}' "
          f"(Confidence: {predictions[top_indices[0]]*100:.1f}%)")
    
    return label_map[top_indices[0]], predictions[top_indices[0]]

print("✅ Các hàm tiền xử lý và dự đoán đã sẵn sàng!")
```

---

## Bước 14: Tạo Ảnh Test (Vẽ Trực Tiếp Trên Colab)

### Cách 1: Tạo ảnh bằng code (Nhanh, dễ test)

```python
# ============================================================
# CELL 15: Tạo ảnh test bằng code
# ============================================================

from PIL import Image, ImageDraw, ImageFont

def create_test_image(character, filename="test_image.png", font_size=150):
    """
    Tạo ảnh nền trắng chữ đen để test model.
    
    Args:
        character: Ký tự muốn tạo (ví dụ: 'A', '5', 'b')
        filename:  Tên file output
        font_size: Kích thước chữ
    """
    # Tạo ảnh trắng 200x200
    img = Image.new('L', (200, 200), color=255)  # 'L' = grayscale, 255 = trắng
    draw = ImageDraw.Draw(img)
    
    # Dùng font mặc định
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Tính vị trí để chữ nằm giữa ảnh
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (200 - text_width) // 2 - bbox[0]
    y = (200 - text_height) // 2 - bbox[1]
    
    # Vẽ chữ đen (0) lên nền trắng (255)
    draw.text((x, y), character, fill=0, font=font)
    
    # Lưu file
    img.save(filename)
    
    # Hiển thị ảnh
    plt.figure(figsize=(3, 3))
    plt.imshow(np.array(img), cmap='gray')
    plt.title(f"Test image: '{character}'", fontsize=14)
    plt.axis('off')
    plt.show()
    
    print(f"✅ Đã tạo ảnh '{filename}' cho ký tự '{character}'")
    return filename

# --- Tạo vài ảnh test ---
test_chars = ['A', '3', 'B', '7', 'R']

for char in test_chars:
    filename = f"test_{char}.png"
    create_test_image(char, filename)
```

### Cách 2: Upload ảnh tự vẽ (Thực tế hơn)

```python
# ============================================================
# CELL 16: Upload ảnh từ máy tính
# ============================================================
from google.colab import files

print("📤 Hãy upload ảnh nền trắng, chữ/số đen:")
print("   - Format: PNG hoặc JPG")
print("   - Nền: TRẮNG")
print("   - Chữ/số: ĐEN")
print("   - Nên viết to, rõ ràng, ở giữa ảnh\n")

uploaded = files.upload()

# Lấy tên file đã upload
uploaded_files = list(uploaded.keys())
print(f"\n✅ Đã upload {len(uploaded_files)} file: {uploaded_files}")
```

> **💡 Mẹo tạo ảnh test tốt:**
> - Dùng Paint (Windows) hoặc bất kỳ app vẽ nào
> - Nền trắng, dùng bút đen, nét to
> - Viết chữ/số **to, ở giữa** ảnh
> - Kích thước khuyến nghị: 200×200 pixels trở lên

---

## Bước 15: Chạy Dự Đoán!

```python
# ============================================================
# CELL 17: DỰ ĐOÁN KÝ TỰ!
# ============================================================

print("🔮 BẮT ĐẦU DỰ ĐOÁN")
print("=" * 50)

# --- Dự đoán ảnh tạo bằng code ---
if 'test_chars' in dir() and test_chars:
    for char in test_chars:
        filename = f"test_{char}.png"
        if os.path.exists(filename):
            print(f"\n--- Dự đoán file: {filename} ---")
            predicted_char, confidence = predict_character(
                filename, model, label_to_char, top_k=5
            )

# --- Dự đoán ảnh upload (nếu có) ---
if 'uploaded_files' in dir() and uploaded_files:
    print("\n\n📤 DỰ ĐOÁN ẢNH UPLOAD:")
    print("=" * 50)
    for file in uploaded_files:
        print(f"\n--- Dự đoán file: {file} ---")
        predicted_char, confidence = predict_character(
            file, model, label_to_char, top_k=5
        )
```

**✅ Output mong đợi:** Mỗi ảnh sẽ hiển thị:
- Ảnh đầu vào bên trái
- Bar chart top-5 dự đoán bên phải
- Kết quả text: `🎯 KẾT QUẢ: Ký tự dự đoán là 'A' (Confidence: 95.3%)`

---

---

# PHASE 7: LƯU & TRIỂN KHAI

---

## Bước 16: Lưu Model

```python
# ============================================================
# CELL 18: Lưu model đã train
# ============================================================

# --- Cách 1: Lưu toàn bộ model (Khuyến nghị) ---
model.save('handwriting_model.keras')
print("✅ Model đã lưu: handwriting_model.keras")

# --- Cách 2: Lưu chỉ weights ---
model.save_weights('handwriting_weights.weights.h5')
print("✅ Weights đã lưu: handwriting_weights.weights.h5")

# --- Download về máy ---
from google.colab import files

files.download('handwriting_model.keras')
print("\n📥 Model đã được download về máy!")
print("   Bạn có thể load lại bằng:")
print("   model = keras.models.load_model('handwriting_model.keras')")
```

---

## Bước 17: Tạo Ứng Dụng Demo Đơn Giản (Canvas Vẽ Tay)

```python
# ============================================================
# CELL 19: Demo - Vẽ tay trên canvas và dự đoán
# ============================================================

# Tạo giao diện vẽ bằng HTML/JS ngay trong Colab
from IPython.display import HTML, display
from google.colab import output

html_code = """
<style>
    .container { text-align: center; font-family: Arial, sans-serif; }
    canvas { border: 2px solid #333; cursor: crosshair; background: white; }
    button { 
        margin: 10px 5px; padding: 10px 25px; font-size: 16px; 
        border: none; border-radius: 5px; cursor: pointer; 
    }
    .btn-predict { background: #2ecc71; color: white; }
    .btn-clear { background: #e74c3c; color: white; }
    .btn-predict:hover { background: #27ae60; }
    .btn-clear:hover { background: #c0392b; }
    h2 { color: #2c3e50; }
    #result { font-size: 24px; margin-top: 15px; color: #2c3e50; }
</style>

<div class="container">
    <h2>✍️ Vẽ Chữ/Số Để Dự Đoán</h2>
    <p>Dùng chuột vẽ chữ hoặc số (nét đen trên nền trắng)</p>
    <canvas id="canvas" width="280" height="280"></canvas>
    <br>
    <button class="btn-predict" onclick="predictDrawing()">🔮 Dự Đoán</button>
    <button class="btn-clear" onclick="clearCanvas()">🗑️ Xóa</button>
    <div id="result"></div>
</div>

<script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    
    // Fill white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing settings
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    canvas.addEventListener('mousedown', (e) => {
        drawing = true;
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (!drawing) return;
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    });
    
    canvas.addEventListener('mouseup', () => drawing = false);
    canvas.addEventListener('mouseout', () => drawing = false);
    
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result').innerHTML = '';
    }
    
    function predictDrawing() {
        const dataUrl = canvas.toDataURL('image/png');
        google.colab.kernel.invokeFunction('predict_from_canvas', [dataUrl], {});
    }
</script>
"""

def predict_from_canvas(data_url):
    """Nhận ảnh từ canvas, xử lý và dự đoán."""
    import base64
    from io import BytesIO
    
    # Decode base64 image
    header, data = data_url.split(',', 1)
    image_data = base64.b64decode(data)
    image = Image.open(BytesIO(image_data)).convert('L')  # Grayscale
    
    # Resize về 28x28
    image = image.resize((28, 28), Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(image).astype('float32')
    
    # Invert (trắng→đen, đen→trắng) vì EMNIST dùng nền đen
    img_array = 255 - img_array
    
    # Normalize
    img_array = img_array / 255.0
    
    # Reshape for model
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx] * 100
    
    # Top 3 predictions
    top3 = predictions.argsort()[-3:][::-1]
    result_text = f"<b>🎯 Dự đoán: '{label_to_char[top_idx]}'</b> ({confidence:.1f}%)<br>"
    result_text += "Top 3: "
    for idx in top3:
        result_text += f"'{label_to_char[idx]}' ({predictions[idx]*100:.1f}%) | "
    
    # Display result
    display(HTML(f"<div style='font-size:20px; text-align:center; margin:10px;'>{result_text}</div>"))

# Register callback
output.register_callback('predict_from_canvas', predict_from_canvas)

# Display canvas
display(HTML(html_code))
```

**✅ Output mong đợi:** Canvas vẽ tay xuất hiện trong Colab → vẽ chữ/số → click "Dự Đoán" → hiển thị kết quả.

---

---

# 📝 TỔNG KẾT

---

## Checklist Hoàn Thành

| # | Bước | Trạng thái |
|---|------|-----------|
| 1 | Tạo Colab, bật GPU | ⬜ |
| 2 | Import thư viện | ⬜ |
| 3 | Tải dataset EMNIST | ⬜ |
| 4 | Tìm hiểu label mapping | ⬜ |
| 5 | Trực quan hóa + Tiền xử lý | ⬜ |
| 6 | Xây dựng CNN model | ⬜ |
| 7 | Compile model | ⬜ |
| 8 | Cấu hình callbacks | ⬜ |
| 9 | Training model | ⬜ |
| 10 | Vẽ biểu đồ training | ⬜ |
| 11 | Đánh giá test set | ⬜ |
| 12 | Confusion matrix | ⬜ |
| 13 | Hàm tiền xử lý ảnh | ⬜ |
| 14 | Tạo ảnh test | ⬜ |
| 15 | Chạy dự đoán | ⬜ |
| 16 | Lưu model | ⬜ |
| 17 | Demo canvas vẽ tay | ⬜ |

---

## Kết Quả Mong Đợi Cuối Cùng

| Metric | Giá trị kỳ vọng |
|--------|-----------------|
| Test Accuracy | **~85-88%** |
| Số classes | **47** (0-9 + A-Z + 11 chữ thường) |
| Thời gian train | **5-10 phút** (với GPU) |
| Model size | **~5-8 MB** |

---

## 🚀 Nâng Cấp Tiếp Theo (Tùy chọn)

Nếu muốn cải thiện thêm, thử:

1. **Data Augmentation**: Xoay, zoom, dịch chuyển ảnh khi train
2. **Deeper model**: Thêm nhiều Conv2D layers
3. **Transfer Learning**: Dùng model pre-trained (ResNet, EfficientNet)
4. **Web App**: Deploy lên Gradio hoặc Streamlit để ai cũng dùng được
5. **Chỉ nhận dạng số**: Dùng MNIST thay vì EMNIST → accuracy ~99%

---

## ❓ Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `ModuleNotFoundError: emnist` | Chưa cài thư viện | Chạy `!pip install emnist` |
| `ResourceExhaustedError` | Hết RAM GPU | Giảm `batch_size` xuống 64 |
| `FileNotFoundError` | Sai đường dẫn ảnh | Kiểm tra tên file, dùng `!ls` để liệt kê |
| Model dự đoán sai | Ảnh chưa xử lý đúng | Kiểm tra invert màu, resize 28x28 |
| Accuracy thấp (<70%) | Train chưa đủ hoặc model nhỏ | Tăng epochs, thêm layers |
| `CUDA out of memory` | GPU hết bộ nhớ | Runtime → Restart runtime, giảm batch_size |

---

> **🎉 Chúc bạn hoàn thành dự án thành công!**
> 
> Mỗi cell code đã có comment giải thích chi tiết.
> Hãy chạy từng cell một, đọc output, hiểu rồi mới chạy cell tiếp theo.
