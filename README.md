# 🌱 CocoaGuard: Intelligent Field Diagnostics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![TFLite](https://img.shields.io/badge/AI_Engine-TFLite-orange.svg)
![Gemini](https://img.shields.io/badge/Cloud_API-Google_Gemini-00a67e.svg)

**CocoaGuard** is a modern, mobile-first precision agriculture web application designed to help farmers and agronomists instantly diagnose cocoa pod diseases. Built with a robust hybrid-AI architecture, it ensures reliable operation whether you are in a high-speed data zone or entirely offline in the field.

---

## ✨ Key Features

* **Hybrid Inference Engine:** 
  * **Offline Mode:** Utilizes a lightweight, custom-trained 22MB `EfficientNetV2-B0` edge model via `ai-edge-litert` for instant, memory-efficient local predictions.
  * **Online Mode:** Automatically detects network connectivity and routes images through the **Google Gemini 2.5 Flash API** for cloud-verified diagnoses.
* **Multilingual Action Plans:** Translates complex agricultural remedy plans into localized languages (Tamil, Malayalam, Hindi, Telugu) in real-time.
* **Visual Confidence Metrics:** Dynamic UI elements that display AI certainty scores and flag "visual overlaps" or fungal complexes (e.g., Phytophthora vs. Monilia) for human review.
* **Auto-Generated Field Reports:** Instantly generate and download timestamped `.txt` diagnostic reports for agricultural record-keeping.

---

## 🛠️ Technology Stack

* **Frontend & Framework:** [Streamlit](https://streamlit.io/) (Mobile-optimized layout)
* **Local AI Engine:** TFLite (`ai-edge-litert`), NumPy, Pillow
* **Cloud AI API:** Google GenAI SDK (`google-genai`)
* **Deployment:** Streamlit Community Cloud (Optimized for low-memory constraints)

---

## 🚀 Run it Locally

If you want to clone this repository and run CocoaGuard on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/sugumaranj/Cocoa-Disease-Classifier/](https://github.com/sugumaranj/Cocoa-Disease-Classifier/)
cd Cocoa-Disease-Classifier
```

**2. Install the required dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your API Key**
Create a hidden `.streamlit` folder and a `secrets.toml` file inside it to securely store your Google Gemini API key.
```bash
mkdir .streamlit
```
Inside `.streamlit/secrets.toml`, add the following line:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

**4. Launch the application**
```bash
streamlit run app.py
```

---

## 👨‍💻 About the Developer

**Developed by Sugumaran J.** 

CocoaGuard bridges the gap between agricultural science and accessible software. Combining a foundational background in Horticulture with computer science application development, this project reflects a commitment to building practical, tech-driven solutions for real-world farming challenges. Designed as a step toward comprehensive precision agriculture and digital coordination systems in the field.
