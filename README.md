# Sign-Language-Translator

A real-time American Sign Language (ASL) recognition system using OpenCV and Deep Learning (CNN with transfer learning) to help translate hand gestures into text and optionally speech. This can assist people with speech or hearing impairments to communicate more effectively.

# 🧠 Model Details

- 📐 Input Size: 128 x 128 x 3 (RGB)
- 🏗️ Base Model: MobileNetV2 (ImageNet weights)
- ⚙️ Layers: Global Avg Pooling → Dense (128) → Dense (26 classes)
- 📉 Loss: Categorical Crossentropy
- 🎯 Optimizer: Adam
- 📊 Epochs: 15 | Batch size: 32
- ✅ Accuracy: ~98% on training | ~94% on validation

---

## 🛠️ Setup Instructions

### 1. Clone this repository

git clone https://github.com/your-username/ASL-Translator.git
cd ASL-Translator
