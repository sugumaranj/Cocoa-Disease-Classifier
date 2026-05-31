# ==============================================================================
# FILE: app.py
# DESCRIPTION: Modern Mobile-First Cocoa Disease Diagnosis Dashboard
#              Features: API Integration, Visual Metrics, & Downloadable Reports
#              Dual-Engine: TFLite (Active for Cloud) / TensorFlow (Commented)
# ==============================================================================

import os
import socket
import datetime
import streamlit as st
import numpy as np
from PIL import Image
from google import genai

# ==============================================================================
# ENGINE SELECTION: TFLite (Active) vs TensorFlow (Preserved in Comments)
# ==============================================================================

# --- ACTIVE: Ultra-Lightweight TFLite Engine (For Free Cloud Deployment) ---
try:
    from ai_edge_litert.interpreter import Interpreter
    AI_ENGINE = "TFLite"
    TF_READY = True
except Exception as e:
    TF_READY = False
    TF_ERROR_MSG = str(e)

# --- PRESERVED: Heavy TensorFlow Engine (For Local PC execution) ---
# try:
#     import tensorflow as tf
#     from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
#     AI_ENGINE = "TensorFlow Keras"
#     TF_READY = True
# except Exception as e:
#     TF_READY = False
#     TF_ERROR_MSG = str(e)

# ==============================================================================
# GOOGLE GEMINI ONLINE API CONFIGURATION
# ==============================================================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------------------------------------------------------------------
# 1. NETWORK HELPER FUNCTION
# ------------------------------------------------------------------------------
def check_internet_connection():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        pass
    return False

# ------------------------------------------------------------------------------
# 2. PAGE CONFIGURATION & MODERN UI STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CocoaGuard 🌱",
    page_icon="🌱",
    layout="centered"
)

st.markdown("""
    <style>
    .main {background-color: #f4f7f6;}
    h1, h2, h3 {color: #1b5e20;}
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        background-color: #2e7d32; 
        color: white;
        height: 3em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {background-color: #1b5e20;}
    .status-badge {
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        display: inline-block;
        margin-bottom: 20px;
    }
    .online {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .offline {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .container-box {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-text { font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 3. HEADER & STATUS DASHBOARD
# ------------------------------------------------------------------------------
st.title("🌱 CocoaGuard")
st.markdown("### Intelligent Field Diagnostics")

if not TF_READY:
    st.error(f"🚨 Critical System Error: Local {AI_ENGINE} Engine failed to initialize.")
    st.code(TF_ERROR_MSG)
    st.stop()

is_online = check_internet_connection()
if is_online:
    st.markdown('<div class="status-badge online">🟢 System Status: ONLINE (Cloud API Ready)</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-badge offline">🔴 System Status: OFFLINE (Using {AI_ENGINE})</div>', unsafe_allow_html=True)

st.info(
    "⚠️ **Diagnostic Scope Limitations:**\n"
    "Our targeted local AI model is strictly calibrated to detect **Anthracnose, CSSVD, Monilia, Phytophthora, Pod Borer, or Healthy pods**. "
    "If your crop suffers from an unlisted disease, the offline AI may misclassify it. "
    "Cloud API mode (Online) provides broader general disease recognition."
)
# ------------------------------------------------------------------------------
# 4. CORE ASSETS: LOCAL MODEL & PREDEFINED REMEDY ENGINE
# ------------------------------------------------------------------------------
@st.cache_resource
def load_local_neural_network():
    # --- ACTIVE: TFLite Model Loader ---
    model_name = "max_efficiency_cocoa_model.tflite" # Assumes model is in the same folder for cloud deployment
    if os.path.exists(model_name):
        try:
            interpreter = Interpreter(model_path=model_name)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            return None
    return None

    # --- PRESERVED: TensorFlow Model Loader ---
    # model_name = os.path.join("..", "notebooks", "v2b0", "max_efficiency_cocoa_model.keras")
    # if os.path.exists(model_name):
    #     try:
    #         return tf.keras.models.load_model(model_name)
    #     except Exception as e:
    #         return None
    # return None

local_model = load_local_neural_network()
CLASS_NAMES = ['Anthracnose', 'CSSVD', 'Healthy', 'Monilia', 'Phytophthora', 'PodBorer']

OFFLINE_REMEDY_DB = {
    "Anthracnose": "Prune infected pods and internal canopy branches immediately to boost airflow. Apply protective copper-based fungicides during heavy rainy periods.",
    "CSSVD": "Cocoa Swollen Shoot Virus Disease has no chemical cure. Completely eradicate, uproot, and burn infected trees to halt vector (mealybug) transmission.",
    "Healthy": "Your cocoa pod shows excellent health profiles. Maintain standard NPK fertilizer distribution arrays and regular scheduled soil moisture irrigation.",
    "Monilia": "Collect, remove, and deeply bury all mummified pods every 7 days to break the fungal spore lifecycle. Prune overhead shade trees to drop humidity.",
    "Phytophthora": "Instantly harvest and destroy black rotten pods. Spray target applications of copper hydroxide or metalaxyl fungicides at the onset of monsoon seasons.",
    "PodBorer": "Enclose developing young pods using biodegradable plastic sleeves. Implement strict harvesting intervals every 7 days to break larval cycles."
}

# ------------------------------------------------------------------------------
# 5. INTERACTIVE CORNER: CAMERA CAPTURE & FILE UPLOADER
# ------------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="container-box">', unsafe_allow_html=True)
    st.markdown("#### 📸 Step 1: Image Capture")
    
    input_mode = st.radio("Select input method:", ["Camera Scanner", "Local File Browser"], horizontal=True)
    uploaded_file = None

    if input_mode == "Camera Scanner":
        camera_img = st.camera_input("Position the cocoa pod clearly in frame:")
        if camera_img: uploaded_file = camera_img
    else:
        file_img = st.file_uploader("Select an image from your device storage:", type=["jpg", "jpeg", "png"])
        if file_img: uploaded_file = file_img

    target_language = st.selectbox(
        "🌍 Select Remedy Language (Requires Online Mode):",
        ["English", "Tamil (தமிழ்)", "Malayalam (മലയാളം)", "Hindi (हिन्दी)", "Telugu (తెలుగు)"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. DATA PROCESSING & HYBRID INFERENCE ENGINE
# ------------------------------------------------------------------------------
if uploaded_file is not None:
    st.markdown("#### 🔬 Step 2: System Evaluation")
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Target Field Sample", use_container_width=True)
    
    predicted_class = None
    confidence_score = 0.0
    second_choice_class = "Healthy"
    api_success = False 

    # ==========================================================================
    # ONLINE CLOUD VISION API
    # ==========================================================================
    if is_online:
        with st.spinner("Analyzing via Google Cloud Vision API..."):
            try:
                prompt_instructions = (
                    "Analyze this cocoa pod image. Identify if it displays symptoms of Anthracnose, "
                    "CSSVD, Monilia, Phytophthora, PodBorer, or if it is completely Healthy. "
                    "Respond with ONLY the exact class name from that list."
                )
                api_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt_instructions, raw_image]
                )
                cleaned_pred = api_response.text.strip()
                if cleaned_pred in CLASS_NAMES:
                    predicted_class = cleaned_pred
                    confidence_score = 100.0 
                    api_success = True
            except Exception as api_error:
                st.warning("📡 Cloud API timeout/error. Instantly falling back to Local Offline Model...")
                api_success = False

    # --------------------------------------------------------------------------
    # LOCAL MODEL INFERENCE PIPELINE
    # --------------------------------------------------------------------------
    if not api_success:
        if local_model is None:
            st.error(f"⚠️ **System Error:** Local {AI_ENGINE} model not found. Check file paths.")
            st.stop()
            
        with st.spinner(f"Processing offline via {AI_ENGINE} neural network..."):
            img_resized = raw_image.resize((224, 224))
            
            # --- ACTIVE: TFLite Inference ---
            img_array = np.array(img_resized, dtype=np.float32)
            img_batch = np.expand_dims(img_array, axis=0)
            
            input_details = local_model.get_input_details()
            output_details = local_model.get_output_details()
            
            local_model.set_tensor(input_details[0]['index'], img_batch)
            local_model.invoke()
            raw_predictions = local_model.get_tensor(output_details[0]['index'])
            
            highest_index = np.argmax(raw_predictions[0])
            predicted_class = CLASS_NAMES[highest_index]
            
            preds = raw_predictions[0]
            if np.max(preds) > 1.0: # Softmax conversion if output is raw logits
                e_x = np.exp(preds - np.max(preds))
                preds = e_x / e_x.sum(axis=0)
                
            confidence_score = preds[highest_index] * 100
            cloned_predictions = preds.copy()
            cloned_predictions[highest_index] = -1.0 
            second_highest_index = np.argmax(cloned_predictions)
            second_choice_class = CLASS_NAMES[second_highest_index]

            # --- PRESERVED: TensorFlow Keras Inference ---
            # img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            # img_batch = np.expand_dims(img_array, axis=0)
            # img_preprocessed = preprocess_input(img_batch)
            
            # raw_predictions = local_model.predict(img_preprocessed)
            # highest_index = np.argmax(raw_predictions[0])
            
            # predicted_class = CLASS_NAMES[highest_index]
            # confidence_score = raw_predictions[0][highest_index] * 100
            
            # cloned_predictions = raw_predictions[0].copy()
            # cloned_predictions[highest_index] = -1.0 
            # second_highest_index = np.argmax(cloned_predictions)
            # second_choice_class = CLASS_NAMES[second_highest_index]

    # ------------------------------------------------------------------------------
    # 7. UPGRADED MATRIX SAFETY THRESHOLDS & VISUAL METRICS
    # ------------------------------------------------------------------------------
    final_verified_diagnosis = predicted_class
    
    st.markdown("---")
    # NEW: Visual Confidence Meter
    st.markdown(f'<div class="metric-text">AI Confidence Score: {confidence_score:.1f}%</div>', unsafe_allow_html=True)
    st.progress(int(confidence_score) / 100.0)
    
    if not api_success:
        if confidence_score < 50.0:
            st.warning(f"⚠️ **Low Confidence / Visual Overlap Suspected**")
            st.info(f"The system leans toward **{predicted_class}**, but has strong secondary indicators of **{second_choice_class}**. Please check both options manually.")
            final_verified_diagnosis = predicted_class if predicted_class != "Healthy" else second_choice_class
        elif predicted_class == "Healthy" and confidence_score < 85.0:
            st.warning(f"⚠️ **Uncertain Clear Signatures Detected**")
            st.info(f"The system flags a tendency toward a clean pod, but underlying indicators display trace features of **{second_choice_class}**. An early infection might be starting.")
            final_verified_diagnosis = second_choice_class 
        elif predicted_class in ["Phytophthora", "Anthracnose", "Monilia"] and confidence_score < 75.0:
            st.warning(f"⚠️ **Fungal Complex Warning**")
            st.info(f"Detected fungal rot symptoms resembling **{predicted_class}**, but due to visual similarities, it could also be **{second_choice_class}**. Monitor pod closely.")
            final_verified_diagnosis = predicted_class
        else:
            if predicted_class == "Healthy":
                st.success(f"✅ **Verified: Healthy Crop Profile**")
            else:
                st.error(f"🚨 **Alert: Confirmed {predicted_class} Infection**")
            final_verified_diagnosis = predicted_class
    else:
        if predicted_class == "Healthy":
            st.success(f"✅ **Cloud Verified: Healthy Crop Profile**")
        else:
            st.error(f"🚨 **Cloud Alert: Confirmed {predicted_class} Infection**")

    # ------------------------------------------------------------------------------
    # 8. TREATMENT EXTRACTION & CLOUD TRANSLATION ENGINE
    # ------------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 Step 3: Targeted Field Remedy Action Plan")
    
    active_remedy_text = OFFLINE_REMEDY_DB[final_verified_diagnosis]
    secondary_remedy_text = None
    if not api_success and confidence_score < 50.0 and second_choice_class in OFFLINE_REMEDY_DB:
        secondary_remedy_text = OFFLINE_REMEDY_DB[second_choice_class]

    # Output text placeholders for the report generator
    report_main_remedy = active_remedy_text
    report_secondary_remedy = secondary_remedy_text

    if target_language == "English":
        st.info(f"**Primary Action Plan ({final_verified_diagnosis}):**\n{active_remedy_text}")
        if secondary_remedy_text:
            st.info(f"**Secondary Precautionary Plan ({second_choice_class}):**\n{secondary_remedy_text}")
    else:
        if is_online:
            with st.spinner(f"Translating to {target_language}..."):
                try:
                    translation_prompt = f"Translate the following agricultural treatment into {target_language}. Keep technical names intact. Text: {active_remedy_text}"
                    report_main_remedy = client.models.generate_content(
                        model='gemini-2.5-flash', contents=translation_prompt
                    ).text
                    st.info(f"**Primary Action Plan ({final_verified_diagnosis}):**\n{report_main_remedy}")
                    
                    if secondary_remedy_text:
                        translation_prompt_sec = f"Translate this into {target_language}: {secondary_remedy_text}"
                        report_secondary_remedy = client.models.generate_content(
                            model='gemini-2.5-flash', contents=translation_prompt_sec
                        ).text
                        st.info(f"**Secondary Precautionary Plan ({second_choice_class}):**\n{report_secondary_remedy}")
                except Exception as translation_error:
                    st.warning("📡 Live Translation failed. Reverting to offline English data:")
                    st.info(f"**Primary Action Plan ({final_verified_diagnosis}):**\n{active_remedy_text}")
                    if secondary_remedy_text:
                        st.info(f"**Secondary Precautionary Plan ({second_choice_class}):**\n{secondary_remedy_text}")
        else:
            st.warning(f"📡 Device Offline. Local language generation ({target_language}) requires internet. Displaying English reference guide below:")
            st.info(f"**Primary Action Plan ({final_verified_diagnosis}):**\n{active_remedy_text}")
            if secondary_remedy_text:
                st.info(f"**Secondary Precautionary Plan ({second_choice_class}):**\n{secondary_remedy_text}")

    # ------------------------------------------------------------------------------
    # 9. GENERATE DOWNLOADABLE REPORT
    # ------------------------------------------------------------------------------
    st.markdown("---")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the text file content
    txt_report = f"""========================================
COCOAGUARD FIELD DIAGNOSIS REPORT
========================================
Date/Time:    {timestamp}
Target Model: EfficientNetV2-B0 (Local/Cloud Hybrid)
----------------------------------------
[DIAGNOSIS RESULTS]
Verified Class : {final_verified_diagnosis}
Confidence     : {confidence_score:.2f}%
Language Output: {target_language}

[PRIMARY ACTION PLAN]
{report_main_remedy}
"""
    if report_secondary_remedy:
        txt_report += f"\n[SECONDARY WARNING PRECAUTION]\n{report_secondary_remedy}\n"
    
    txt_report += "========================================\nSystem built for precision agriculture."

    # Download button widget
    st.download_button(
        label="📥 Download Field Report (.txt)",
        data=txt_report,
        file_name=f"CocoaGuard_Report_{final_verified_diagnosis}.txt",
        mime="text/plain"
    )
