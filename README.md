# 🎭 Hybrid Deepfake Detection System

A Hybrid Deepfake Detection System using **Deep Learning** to identify manipulated images and videos.  
This project combines **Xception + BiLSTM + Attention Mechanism** for accurate deepfake classification and provides an interactive **Streamlit-based interface**.

---

## 📌 Project Overview

Deepfakes are AI-generated manipulated media that can create realistic fake videos and images.  
This project aims to detect such forged media using a hybrid deep learning architecture.

The system:

- Detects fake and real videos/images
- Extracts facial and temporal features
- Uses deep learning for classification
- Displays prediction results using a Streamlit interface
- Stores prediction history and generates evaluation plots

---

## 🚀 Features

✅ Deepfake detection using Deep Learning  
✅ Hybrid Architecture (Xception + BiLSTM + Attention)  
✅ Streamlit web interface  
✅ Image and Video support  
✅ Prediction history tracking  
✅ Accuracy and ROC evaluation plots  
✅ Real vs Fake classification  

---

## 🧠 Model Architecture

The proposed model combines:

### 1. Xception Network
- Extracts spatial facial features
- Captures forgery artifacts

### 2. BiLSTM
- Learns temporal dependencies across video frames
- Helps analyze motion inconsistencies

### 3. Attention Mechanism
- Focuses on suspicious frames
- Improves classification performance

---

## 📂 Project Structure

```text
HybridDeepfakeDetection/
│
├── app.py
├── train.py
├── evaluate.py
├── database.py
├── generate_plots.py
├── requirements.txt
│
├── models/
├── training/
├── checkpoints/
├── Results/
├── notebooks/
├── utils/
└── data/
```

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Streamlit
- Matplotlib
- Deep Learning

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/Udai0712/HybridDeepfakeDetection.git
cd HybridDeepfakeDetection
```

### Create Virtual Environment

```bash
python -m venv deepfake_env
```

Activate:

Mac/Linux:

```bash
source deepfake_env/bin/activate
```

Windows:

```bash
deepfake_env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

Launch Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Results

The system generates:

- Training Accuracy Graph
- ROC Curve
- Prediction History
- Real vs Fake Prediction Results

Results are stored in the **Results/** folder.

---

## 📸 Sample Output

Add screenshots here:

- System Interface
- Prediction Result
- ROC Curve
- Training Accuracy Graph

---

## 🎯 Applications

- Social media verification
- Fake news detection
- Digital media authentication
- Cybersecurity
- Forensic investigation

---

## 🔮 Future Improvements

- Real-time webcam detection
- Mobile deployment
- Larger dataset training
- Explainable AI visualization
- Cloud deployment

---

## 👨‍💻 Author

**Udai Kiran C**  
M.Tech Integrated CSE  
Vellore Institute of Technology

---

## 📜 License

This project is licensed under the MIT License.