# HƯỚNG DẪN CHẠY TRÊN MÁY LOCAL

## 🎯 Tại sao chạy local?
- Codespace **timeout** khi xử lý dataset lớn (230M rows)
- CPU local **không giới hạn thời gian** chạy
- Tránh bị kill process (Signal 143)

---

## 📥 BƯỚC 1: Clone/Download Code

### Option A: Clone từ GitHub
```bash
git clone https://github.com/quang25808600233-max/cnn---tfm.git
cd cnn---tfm
```

### Option B: Download ZIP
1. Vào GitHub repo: https://github.com/quang25808600233-max/cnn---tfm
2. Click **Code** → **Download ZIP**
3. Giải nén vào thư mục làm việc

---

## 🐍 BƯỚC 2: Setup Python Environment

### Yêu cầu:
- **Python 3.10+** (khuyến nghị 3.12)
- **16GB RAM** trở lên (32GB tối ưu)
- **50GB disk** trống cho chunks

### Cài đặt dependencies:

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS/Linux)
source .venv/bin/activate

# Cài packages
pip install -r requirements.txt
```

**File `requirements.txt` đã có sẵn:**
```
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=5.0.0
tensorflow>=2.20.0
scikit-learn>=1.3.0
```

---

## 📦 BƯỚC 3: Chuẩn bị Data

### Copy file XAUUSD.parquet vào thư mục gốc:

**Windows:**
```bash
copy "C:\Users\Dinh Vuong Ng\Desktop\XAUUSD.parquet" .
```

**macOS/Linux:**
```bash
cp ~/Desktop/XAUUSD.parquet .
```

### Kiểm tra file:
```bash
ls -lh XAUUSD.parquet
# Phải thấy: ~2.3GB, 230M rows
```

---

## 🚀 BƯỚC 4: Chạy Chunking

### Full dataset (230M rows → ~460 chunks):

```bash
python xauusd_chunking_local.py
```

**Thời gian ước tính:**
- **CPU Intel i5/i7**: 6-12 giờ
- **CPU AMD Ryzen 5/7**: 5-10 giờ
- **Apple M1/M2**: 4-8 giờ

**Kết quả:**
- Thư mục `XAUUSD_Chunks/` với ~460 files `.npz`
- Mỗi chunk ~26MB, tổng ~12GB
- File `manifest.json` chứa metadata

### Test với 1 row group (nhanh):
```bash
# Sửa trong xauusd_chunking_local.py:
# 'max_row_groups': 1

python xauusd_chunking_local.py
```

---

## 🧠 BƯỚC 5: Train Model

```bash
python train_model.py
```

**Train settings:**
- **Batch size**: 512 (giảm xuống 256 nếu thiếu RAM)
- **Epochs**: 50 (early stopping enabled)
- **Validation split**: 20%

**Outputs:**
- `best_model.keras` - Model tốt nhất
- `training_history.json` - Metrics theo epoch

---

## ⚙️ Tùy chỉnh CONFIG

### xauusd_chunking_local.py:

```python
CONFIG = {
    'chunk_rows': 20000,      # Tăng lên 50000 nếu RAM nhiều
    'sequence_length': 60,    # Độ dài sequence
    'score_threshold': 1.5,   # Threshold labeling
    'max_row_groups': None,   # None = full dataset
}
```

### train_model.py:

```python
BATCH_SIZE = 512       # Giảm xuống 256 nếu OOM
MAX_CHUNKS = None      # None = train all chunks
EPOCHS = 50
```

---

## 🔍 Monitor Progress

### Trong terminal khác:

**Windows (PowerShell):**
```powershell
Get-Content XAUUSD_Chunks\manifest.json
dir XAUUSD_Chunks\*.npz | Measure-Object
```

**macOS/Linux:**
```bash
watch -n 5 'ls -1 XAUUSD_Chunks/*.npz | wc -l'
tail -f chunking.log  # Nếu redirect output
```

---

## ❌ Troubleshooting

### 1. **ImportError: numpy không tìm thấy**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. **MemoryError khi chunking**
- Giảm `chunk_rows` xuống 10000-15000
- Close các app khác để free RAM

### 3. **Training bị OOM**
- Giảm `BATCH_SIZE` từ 512 → 256 → 128
- Set `MAX_CHUNKS = 100` để train subset

### 4. **File XAUUSD.parquet không tìm thấy**
```bash
# Kiểm tra đường dẫn
ls -la XAUUSD.parquet

# Nếu ở folder khác, sửa CONFIG:
'parquet_path': '/full/path/to/XAUUSD.parquet'
```

---

## 📊 Kiểm tra kết quả

### Sau khi chunking:
```python
# check_chunks.py
import numpy as np
import os

chunks = sorted([f for f in os.listdir('XAUUSD_Chunks') if f.endswith('.npz')])
print(f"Total chunks: {len(chunks)}")

# Load 1 chunk để test
data = np.load(f'XAUUSD_Chunks/{chunks[0]}')
print(f"X shape: {data['X'].shape}")
print(f"y shape: {data['y'].shape}")
print(f"Label distribution: {np.bincount(data['y'])}")
```

### Sau khi training:
```python
# test_model.py
from tensorflow import keras
import numpy as np

model = keras.models.load_model('best_model.keras')
print(model.summary())

# Test predict
data = np.load('XAUUSD_Chunks/chunk_0000.npz')
X_test = data['X'][:100]
preds = model.predict(X_test)
print(f"Predictions: {preds[:5]}")
```

---

## 💡 Tips tối ưu

1. **Run overnight** - Để máy chạy qua đêm
2. **Disable sleep** - Tắt chế độ ngủ trong settings
3. **Close apps** - Đóng Chrome, VSCode để free RAM
4. **SSD preferred** - Chạy trên SSD nhanh hơn HDD
5. **Monitor temperature** - Kiểm tra nhiệt độ CPU (dùng HWMonitor)

---

## 📞 Support

Nếu gặp lỗi:
1. Check terminal output
2. Kiểm tra file `manifest.json` đã tạo chưa
3. Xem log errors trong Python traceback
4. GitHub Issues: https://github.com/quang25808600233-max/cnn---tfm/issues

---

**✅ Sẵn sàng chạy full dataset trên máy local!**
