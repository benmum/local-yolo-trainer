# 🚀 Local YOLO Trainer

A fully local web application for training **Ultralytics YOLO** models with a clean, modern UI.  
Train, monitor, visualize, and export YOLO models directly on your machine — no cloud uploads, notebooks, or complex CLI commands required.

Runs entirely on **localhost** using Flask.

---

## ✨ Features

- 🧠 **YOLO Training UI**
  - Supports Ultralytics YOLO (v8 → v11)
  - Model sizes: nano → xlarge
  - Optional custom model weights or paths

- 📁 **Dataset Support**
  - Local datasets via `data.yaml`
  - Built-in Ultralytics presets (`coco8.yaml`, `coco128.yaml`, etc.)
  - Fully offline — datasets never leave your machine

- 📊 **Live Training Dashboard**
  - Real-time logs streamed to the browser
  - Epoch and percentage progress tracking
  - Clear training status indicators

- 📈 **Metrics & Visualization**
  - Automatically parses `results.csv`
  - Shows precision, recall, mAP@50, mAP@50-95
  - Click-to-enlarge plots (confusion matrix, PR curves, results.png)
  - Dropdown to switch between training runs

- 💾 **Model Export**
  - Save trained `.pt` files anywhere on your PC
  - User-defined model names
  - Download enabled only after training completes

- 🎨 **Modern UI**
  - Dark, vibrant, glass-style interface
  - Beginner-friendly metric explanations
  - Designed for clarity and usability

---

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, Vanilla JavaScript
- **ML Framework:** Ultralytics YOLO
- **Runtime:** Localhost (no external services)

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/benmum/local-yolo-trainer.git
cd local-yolo-trainer
```

### 2️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```


---

## ▶️ Running the App

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 📁 Dataset Format (YOLO Detection)

Your dataset should follow this structure:

```
dataset/
├─ data.yaml
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

Example `data.yaml`:

```yaml
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

> Labels must be in YOLO format:  
> `class x_center y_center width height` (normalized 0–1)

---

## 📊 Metrics Explained

- **Precision** – How many predicted objects were correct  
- **Recall** – How many real objects were detected  
- **mAP@50** – Mean Average Precision at 50% IoU (easier threshold)  
- **mAP@50-95** – Average precision across IoU 0.50–0.95 (stricter, more realistic)

Lower loss values indicate better training quality.

---

## 🚦 Use Cases

- Learning object detection
- Rapid dataset experimentation
- Local model prototyping
- Teaching YOLO visually
- Privacy-sensitive workflows

---

## ⚠️ Notes

- Designed for **local use only**
- Training speed depends on your hardware (CPU/GPU)
- Large datasets and models can consume significant disk space

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it.

---

## 🙌 Acknowledgements

- Ultralytics YOLO

---

## ⭐ Future Ideas

- Run comparison view
- Searchable run history
- Training cancellation
- Desktop packaging (`.exe` / `.app`)
- Additional task support (segmentation, classification)

---

If you find this project useful, consider starring the repo ⭐

---
---

🚧 **More functionality coming soon!**

Planned improvements include:
- Choosing training **optimizers** (SGD, Adam, AdamW, etc.)
- Custom **learning rate**, momentum, and weight decay
- Additional training hyperparameters exposed in the UI
- Expanded task support and enhanced run comparisons
- More visualizations and overall workflow refinements
