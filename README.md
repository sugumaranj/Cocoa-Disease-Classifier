# 🌱 CocoaGuard AI: Enterprise Agricultural Diagnostics Dashboard

[![Open in Streamlit](https://img.shields.io/badge/Open%20in%20Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cocoa-disease-classifier.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![TFLite](https://img.shields.io/badge/AI_Engine-TFLite-orange.svg)
![Gemini](https://img.shields.io/badge/Cloud_API-Google_Gemini-00a67e.svg)
![OpenWeatherMap](https://img.shields.io/badge/API-OpenWeatherMap-eb6e4b.svg)

**CocoaGuard AI** is a modern, mobile-first precision agriculture web application engineered specifically for cocoa farmers. Built with a fault-tolerant, dual-engine AI architecture, it bridges the gap between advanced deep learning and practical agronomy. It shifts farming from a reactive process to a proactive one by combining instant disease diagnosis with real-time, weather-driven fungal alerts.

---

## ✨ Key Features

* **Resilient Dual-Engine Diagnostics:**
  * **Cloud Engine (Default):** Utilizes the powerful **Google Gemini 2.5 Flash API** for highly detailed, generative diagnostic reports.
  * **Edge Failover (Offline-Ready):** If the external API drops, hits a rate limit, or times out, the backend instantly reroutes the image to a locally hosted, custom-trained `EfficientNetV2-B0` model via `ai_edge_litert`. This guarantees 100% diagnostic uptime.
* **Proactive Weather & Fungal Alerts:** Integrates the OpenWeatherMap API to track live temperature and humidity based on the farm's location. Triggers preemptive "Critical Fungal Threat" warnings when humidity spikes above 80% during rainy conditions.
* **Multilingual "Cocoa Doctor" Chatbot:** A built-in generative AI agronomist that translates expert botanical terms, cultural practices, and chemical remedy dosages into regional languages (**Tamil, Telugu, Hindi, Malayalam, and English**).
* **Batch Analytics & Auto-Generated Field Reports:** Process multiple field images simultaneously and download a timestamped `.txt` summary report for agricultural record-keeping.
* **Dynamic Safety Override:** If the local offline model predicts a crop is "Healthy" but the statistical confidence is below 90%, the system automatically flags the next most likely hidden infection to prevent dangerous false negatives.

---

## ⚠️ Diagnostic Scope & ML Pipeline

To ensure responsible agricultural practices, CocoaGuard strictly defines its operational boundaries:
* **Targeted Dataset Scope:** The local TFLite edge model (`max_efficiency_cocoa_model.tflite`) is explicitly calibrated to detect only six specific profiles: **Anthracnose, CSSVD, Monilia, Phytophthora, Pod Borer,** and **Healthy**.
* **Handling Class Imbalance:** The model was trained using a custom **Logarithmic Capped Class Weight** formula to prevent the AI from ignoring rare diseases in the dataset. 
* **Performance:** Evaluated over 1,729 test images, the custom `EfficientNetV2-B0` edge model achieved an overall validation accuracy of >75%, with an exceptionally high precision rate (91.3%) for identifying 'Healthy' crops, establishing a highly reliable baseline.

---

## 🛠️ Technology Stack

* **Frontend & UI:** [Streamlit](https://streamlit.io/) (Mobile-optimized layout), Custom CSS
* **Backend Pipeline:** Python 3, Concurrent Futures (Timeout Management)
* **Local Edge AI:** TensorFlow, Keras, TFLite (`ai_edge_litert`), NumPy, Pillow (Image Normalization pipeline)
* **Cloud APIs:** Google Gemini 2.5 Flash API, OpenWeatherMap API

---

## 🚀 Run it Locally

If you want to clone this repository and run CocoaGuard AI on your own machine or local server, follow these exact steps:

**1. Clone the repository**
```bash
git clone [https://github.com/sugumaranj/Cocoa-Disease-Classifier.git](https://github.com/sugumaranj/Cocoa-Disease-Classifier.git)
cd Cocoa-Disease-Classifier
```
**2. Install the required dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API Key**
Create a hidden `.streamlit` folder and a `secrets.toml` file inside it to securely store your Google Gemini and OpenWeatherMap API key.
```bash
mkdir .streamlit
```
Inside `.streamlit/secrets.toml`, add the following line:
```toml
GEMINI_API_KEY = "your_api_key_here"
WEATHER_API_KEY = "your_openweathermap_api_key_here"
```

**4. Launch the application**
```bash
streamlit run app.py
```

---

## 👨‍💻 About the Developer

**Developed by Sugumaran J** 

CocoaGuard AI bridges the gap between agricultural science and accessible software. Combining a foundational background with a Diploma in Horticulture and ongoing studies in Computer Applications (BCA), this project reflects a commitment to building practical, tech-driven solutions for real-world farming challenges. This platform serves as a foundational step toward a comprehensive, offline-capable precision agriculture system.
