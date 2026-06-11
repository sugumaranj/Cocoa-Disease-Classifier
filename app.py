# ==============================================================================
# FILE: app.py
# DESCRIPTION: Mobile-First Cocoa Disease Diagnosis Dashboard
#              Features: 
#              - Dual-Engine Architecture (Cloud API + Edge TFLite Fallback)
#              - Pessimistic Safety Override (Prevents False 'Healthy' scans)
#              - Session State Memory for UI Stability
#              - Dynamic Theme UI (Auto Light/Dark Mode Compatibility)
# ==============================================================================

import os
import socket
import datetime
import time 
import concurrent.futures 
import streamlit as st
import numpy as np
from PIL import Image
from google import genai

# ==============================================================================
# 1. AI ENGINE INITIALIZATION
# ==============================================================================
# Attempt to load the local TensorFlow Lite model for offline edge processing.
try:
    from ai_edge_litert.interpreter import Interpreter
    AI_ENGINE = "TFLite"
    TF_READY = True
except Exception as e:
    TF_READY = False
    TF_ERROR_MSG = str(e)

# Securely load the Google Gemini API key from Streamlit secrets.
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

# ==============================================================================
# 2. AGRONOMY DATABASE & CONSTANTS
# ==============================================================================
CLASS_NAMES = ['Anthracnose', 'CSSVD', 'Healthy', 'Monilia', 'Phytophthora', 'PodBorer']

# Map technical model output names to simple, farmer-friendly terminology.
SIMPLE_NAMES = {
    "Anthracnose": "Anthracnose (Fungal Spots)",
    "CSSVD": "Cocoa Swollen Shoot Virus (CSSVD)",
    "Healthy": "Healthy Crop",
    "Monilia": "Frosty Pod Rot",
    "Phytophthora": "Black Pod Rot",
    "PodBorer": "Cocoa Pod Borer (Pest)"
}

# Offline knowledge base utilized when the Cloud API is unreachable.
LOCAL_REMEDY_DB = {
    "Anthracnose": {
        "priority": "HIGH",
        "symptoms": "- Sunken dark lesions on pods.\n- Pinkish spores appear in wet conditions.",
        "cultural": "- Prune shade trees to increase airflow.\n- Remove and burn infected pods immediately.",
        "organic": "- Apply Neem oil extract.\n- Use Trichoderma-based biological fungicides.",
        "chemical": "- Spray Copper Oxychloride (50% WP) @ 2.5g/L during rains."
    },
    "CSSVD": {
        "priority": "CRITICAL",
        "symptoms": "- Swollen stems and rounded, small pods.\n- Red vein banding on leaves.",
        "cultural": "- Uproot and burn infected trees immediately.\n- Plant barrier crops to stop mealybug spread.",
        "organic": "- Release natural predators (ladybugs) to control mealybug vectors.",
        "chemical": "- No chemical cure for the virus.\n- Use mild insecticides to control mealybug vectors if needed."
    },
    "Healthy": {
        "priority": "NONE",
        "symptoms": "- Normal pod development.\n- No visible spots, rot, or lesions.",
        "cultural": "- Maintain proper field sanitation.\n- Ensure good soil drainage.",
        "organic": "- Apply organic compost to maintain soil microbiome.",
        "chemical": "- Apply standard NPK fertilizer.\n- No fungicidal intervention needed."
    },
    "Monilia": {
        "priority": "HIGH",
        "symptoms": "- Premature ripening.\n- Thick, white/cream 'frosty' fungal spores on pod.",
        "cultural": "- Remove and bury mummified pods weekly.\n- Reduce overhead canopy shade.",
        "organic": "- Use biocontrol agents like Bacillus subtilis.",
        "chemical": "- Spray Copper Hydroxide (77% WP) @ 2g/L on developing pods."
    },
    "Phytophthora": {
        "priority": "CRITICAL",
        "symptoms": "- Hard, dark black spots.\n- Rot rapidly spreads to engulf the entire pod.",
        "cultural": "- Improve soil drainage trenches.\n- Harvest frequently; do not leave empty husks.",
        "organic": "- Apply Trichoderma viride enriched compost around the base.",
        "chemical": "- Spray Metalaxyl (8% WP) + Mancozeb (64% WP) @ 2.5g/L."
    },
    "PodBorer": {
        "priority": "HIGH",
        "symptoms": "- Premature patchy yellowing.\n- Tiny insect exit holes on the pod surface.",
        "cultural": "- Enclose young pods in biodegradable sleeves.\n- Harvest strictly every 7 days.",
        "organic": "- Use pheromone traps to disrupt mating.\n- Apply Neem-based sprays (Azadirachtin).",
        "chemical": "- Spray Cypermethrin (10% EC) @ 1ml/L safely away from harvest."
    }
}

# ==============================================================================
# 3. UTILITY FUNCTIONS & CACHING
# ==============================================================================
def check_internet_connection():
    """Validates if the device has an active internet connection for the Cloud API."""
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        pass
    return False

def fetch_gemini_diagnosis(prompt, img):
    """Executes the Cloud API call to Gemini."""
    response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, img])
    return response.text.strip()

def clean_api_text(text, unwanted_prefix):
    """Strips redundant headers returned by the LLM to ensure clean UI formatting."""
    cleaned = text.replace(unwanted_prefix, "").replace(unwanted_prefix.lower(), "").strip()
    if not cleaned.startswith("-") and not cleaned.startswith("•"):
        cleaned = "- " + cleaned
    return cleaned.lstrip(":").strip()

@st.cache_data
def translate_text(text, target_lang):
    """Caches translations in memory to prevent repetitive, costly API calls."""
    if target_lang == "English" or not text or client is None: 
        return text
    try:
        return client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"Translate this agricultural text to {target_lang}. Keep chemical names intact and maintain formatting.\n\n{text}"
        ).text.strip()
    except:
        return text

@st.cache_resource
def load_local_neural_network():
    """Loads the TFLite model into memory once to optimize performance."""
    if os.path.exists("max_efficiency_cocoa_model.tflite"):
        try:
            interpreter = Interpreter(model_path="max_efficiency_cocoa_model.tflite")
            interpreter.allocate_tensors()
            return interpreter
        except Exception:
            return None
    return None

local_model = load_local_neural_network()

# ==============================================================================
# 4. APP CONFIGURATION & DYNAMIC CSS
# ==============================================================================
st.set_page_config(page_title="CocoaGuard 🌱", page_icon="🌱", layout="centered")

# Custom CSS utilizes CSS Variables (var(--...)) to natively support Light/Dark Modes.
st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 12px; background-color: #ff9800; color: white;
        height: 3.5em; font-weight: 900; font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.4); border: 2px solid #f57c00; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #e65100; transform: scale(1.02);}
    
    .status-badge {
        padding: 6px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8em;
        display: inline-block; margin-bottom: 10px; text-align: center; width: 100%;
    }
    .api-badge {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .local-badge {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    
    .card {
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        padding: 20px; border-radius: 12px;
        border: 1px solid var(--border-color); margin-bottom: 20px;
    }
    
    .disclaimer-box {
        background-color: rgba(255, 193, 7, 0.15); 
        color: var(--text-color); padding: 12px; 
        border-radius: 8px; border-left: 5px solid #ffc107; margin-bottom: 15px;
        font-size: 0.9em;
    }
    
    .priority-CRITICAL { color: #d32f2f; font-weight: 900; background-color: #f8d7da; padding: 4px 8px; border-radius: 4px;}
    .priority-HIGH { color: #ff9800; font-weight: 900; background-color: rgba(255, 152, 0, 0.2); padding: 4px 8px; border-radius: 4px;}
    .priority-LOW { color: #0c5460; font-weight: 900; background-color: #d1ecf1; padding: 4px 8px; border-radius: 4px;}
    .priority-NONE { color: #155724; font-weight: 900; background-color: #d4edda; padding: 4px 8px; border-radius: 4px;}
    </style>
""", unsafe_allow_html=True)

# Initialize Session State memory. This prevents the app from clearing results 
# during UI interactions.
if "results" not in st.session_state:
    st.session_state.results = []
if "batch_analytics" not in st.session_state:
    st.session_state.batch_analytics = {}

# ==============================================================================
# 5. UI: SIDEBAR NAVIGATION & SETTINGS
# ==============================================================================
is_online = check_internet_connection() and client is not None

with st.sidebar:
    st.header("🌐 System Status")
    if is_online:
        st.markdown('<div class="status-badge api-badge">🟢 ONLINE (Cloud API)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge local-badge">🔴 OFFLINE ({AI_ENGINE})</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    target_language = st.selectbox(
        "🌍 Output Language:",
        ["English", "Tamil (தமிழ்)", "Malayalam (മലയാളം)", "Hindi (हिन्दी)", "Telugu (తెలుగు)"]
    )
    
    st.markdown("---")
    st.header("📝 Farm Details")
    global_farm_name = st.text_input("Farm Owner Name (Optional)")
    global_location = st.text_input("Farm Location (Optional)")
    
    st.markdown("---")
    with st.expander("⚙️ Advanced Settings (Local AI Only)"):
        st.info("If the local AI confidence falls below this threshold, it overrides 'Healthy' to warn you of potential hidden threats.")
        safety_margin = st.slider("Safety Override Threshold (%)", min_value=70, max_value=99, value=90, step=1)

# ==============================================================================
# 6. UI: MAIN HOMEPAGE & IMAGE INPUT
# ==============================================================================
st.title("🌱 CocoaGuard")
st.markdown("### Intelligent Field Diagnostics")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📸 Step 1: Capture or Upload Images")

input_mode = st.radio(
    "Select Input Method:", 
    ["Local File Upload / Phone Camera 📂", "Built-in Web Camera 📷"], 
    horizontal=True
)

uploaded_files = []
if input_mode == "Built-in Web Camera 📷":
    camera_img = st.camera_input("Take a clear picture of the pod/leaf:")
    if camera_img: uploaded_files.append(camera_img)
else:
    st.info("📱 **Mobile Users:** Tap 'Browse files' and select 'Take Photo' to use your phone's native camera with zoom!")
    files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files: uploaded_files.extend(files)

st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 7. CORE DIAGNOSTIC ENGINE (Analyzes and saves to memory)
# ==============================================================================
if uploaded_files:
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        btn_text = "🚀 RUN DIAGNOSTIC" if len(uploaded_files) == 1 else f"🚀 PROCESS {len(uploaded_files)} IMAGES"
        run_btn = st.button(btn_text)

    if run_btn:
        # Clear previous session data for a fresh analysis
        st.session_state.results = []
        st.session_state.batch_analytics = {"Healthy Crop": 0, "Anthracnose (Fungal Spots)": 0, "Cocoa Swollen Shoot Virus (CSSVD)": 0, "Frosty Pod Rot": 0, "Black Pod Rot": 0, "Cocoa Pod Borer (Pest)": 0}
        
        progress_bar = st.progress(0)

        for idx, file in enumerate(uploaded_files):
            raw_image = Image.open(file)
            
            api_success = False 
            diagnosis_source = ""
            final_disease_name = ""
            final_confidence = 0.0
            threat_priority = "NONE"
            top_preds = []
            symp, cult, org, chem = "", "", "", ""

            # ------------------------------------------------------------------
            # STRATEGY A: PRIMARY CLOUD API (Gemini)
            # ------------------------------------------------------------------
            if is_online:
                # Enforce a short sleep on batch uploads to respect free-tier API rate limits
                if idx > 0: time.sleep(2.5) 
                
                with st.spinner("Analyzing via Cloud API..."):
                    prompt = (
                        "Analyze this cocoa crop image. Identify the disease (or state if Healthy). "
                        "Keep answers VERY SHORT. Use bullet points (-) for every new sentence. "
                        "Do NOT mention any universities or research stations. "
                        "Do NOT include header text inside the sections (like 'Symptoms:'). "
                        "For Chemical Control, you MUST extract and state the EXACT active chemical ingredients and EXACT dosage ratios. If no chemical is needed, state 'None'. "
                        "Format EXACTLY using the pipe (|) character with 7 sections:\n"
                        "Disease Name|Confidence %|Priority Level (CRITICAL, HIGH, MODERATE, LOW, NONE)|Symptoms Bullets|Cultural Control Bullets|Organic Control Bullets|Chemical Control Bullets"
                    )
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(fetch_gemini_diagnosis, prompt, raw_image)
                            raw_text = future.result(timeout=60)
                        
                        if "|" in raw_text:
                            parts = raw_text.split("|")
                            if len(parts) >= 7:
                                final_disease_name = parts[0].strip()
                                try: final_confidence = float(parts[1].replace("%","").strip())
                                except: final_confidence = 95.0 
                                
                                threat_priority = parts[2].strip().upper()
                                symp = clean_api_text(parts[3], "Symptoms:")
                                cult = clean_api_text(parts[4], "Cultural Control:")
                                org = clean_api_text(parts[5], "Organic Control:")
                                chem = clean_api_text(parts[6], "Chemical Control:")
                                
                                api_success = True
                                diagnosis_source = "Cloud API 🟢"
                    except:
                        api_success = False

            # ------------------------------------------------------------------
            # STRATEGY B: EDGE FALLBACK (TFLite Local Model)
            # ------------------------------------------------------------------
            if not api_success:
                if local_model is None:
                    st.error("System Error: Local offline model unavailable.")
                    continue
                    
                diagnosis_source = "Local AI Model 🔴"
                with st.spinner("Processing via Local Edge AI..."):
                    
                    # Pre-process image for the TensorFlow Lite interpreter
                    img_resized = raw_image.resize((224, 224))
                    img_array = np.array(img_resized, dtype=np.float32)
                    img_batch = np.expand_dims(img_array, axis=0)
                    
                    input_details = local_model.get_input_details()
                    output_details = local_model.get_output_details()
                    local_model.set_tensor(input_details[0]['index'], img_batch)
                    local_model.invoke()
                    raw_predictions = local_model.get_tensor(output_details[0]['index'])[0]
                    
                    # Softmax calculation to convert output to percentages
                    if np.max(raw_predictions) > 1.0: 
                        probs = (np.exp(raw_predictions - np.max(raw_predictions)) / np.exp(raw_predictions - np.max(raw_predictions)).sum()) * 100
                    else:
                        probs = raw_predictions * 100
                        
                    sorted_indices = np.argsort(probs)[::-1]
                    highest_index = sorted_indices[0]
                    raw_predicted_class = CLASS_NAMES[highest_index]
                    final_confidence = probs[highest_index]
                    
                    # PESSIMISTIC SAFETY OVERRIDE: Intercept rigid model bias toward false 'Healthy' readings
                    if raw_predicted_class == "Healthy" and final_confidence < safety_margin:
                        non_healthy_indices = [i for i in sorted_indices if CLASS_NAMES[i] != "Healthy"]
                        highest_index = non_healthy_indices[0]
                        raw_predicted_class = CLASS_NAMES[highest_index]
                        final_confidence = probs[highest_index]
                        top_preds.append(("Safety Override Triggered", 0.0, "N/A")) 

                    final_disease_name = SIMPLE_NAMES[raw_predicted_class]
                    threat_priority = LOCAL_REMEDY_DB[raw_predicted_class]["priority"]
                    
                    db_entry = LOCAL_REMEDY_DB[raw_predicted_class]
                    symp, cult, org, chem = db_entry['symptoms'], db_entry['cultural'], db_entry['organic'], db_entry['chemical']

                    # Capture alternative predictions if confidence is critically low
                    if final_confidence < 70.0 and len(top_preds) == 0:
                        for i in range(min(2, len(sorted_indices))):
                            c_name = CLASS_NAMES[sorted_indices[i]]
                            top_preds.append((SIMPLE_NAMES[c_name], probs[sorted_indices[i]], c_name))

            # Push results to session state memory
            st.session_state.results.append({
                "image": raw_image,
                "disease": final_disease_name,
                "confidence": final_confidence,
                "priority": threat_priority,
                "symptoms": symp,
                "cultural": cult,
                "organic": org,
                "chemical": chem,
                "source": diagnosis_source,
                "top_preds": top_preds
            })

            # Update batch analytics metrics
            if final_disease_name in st.session_state.batch_analytics:
                st.session_state.batch_analytics[final_disease_name] += 1
            else:
                st.session_state.batch_analytics[final_disease_name] = 1
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        st.success("✅ Analysis Complete!")

# ==============================================================================
# 8. RENDER RESULTS (Interactive Memory Phase)
# ==============================================================================
if st.session_state.get("results"):
    
    # Initialize the downloadable report array
    report_lines = []
    report_lines.append("========================================\n   COCOAGUARD FIELD DIAGNOSIS REPORT\n========================================")
    report_lines.append(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Farm Owner: {global_farm_name or 'Not Provided'}")
    report_lines.append(f"Location: {global_location or 'Not Provided'}")
    report_lines.append("\n[BATCH ANALYTICS SUMMARY]")
    for k, v in st.session_state.batch_analytics.items():
        if v > 0: report_lines.append(f"- {k}: {v} image(s)")
    report_lines.append("----------------------------------------\n")

    # Render UI cards for each processed image
    for idx, res in enumerate(st.session_state.results):
        st.markdown(f"### 🔍 Analysis: Image {idx+1}")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Display Image Thumbnail
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2: st.image(res["image"], caption=f"Image {idx+1}", use_container_width=True)
        
        # Display Core Metrics
        st.markdown(f"**Data Source:** {res['source']}")
        st.subheader(f"Identified: {res['disease']}")
        
        priority_class = f"priority-{res['priority']}" if res['priority'] in ["CRITICAL", "HIGH", "LOW", "NONE"] else "priority-HIGH"
        st.markdown(f'**Threat Priority:** <span class="{priority_class}">{res['priority']}</span>', unsafe_allow_html=True)
        st.progress(int(res['confidence']) / 100.0)
        st.markdown(f"**Confidence Score:** {res['confidence']:.1f}%")

        # Display Contextual Warnings
        if res['top_preds'] and res['top_preds'][0][0] == "Safety Override Triggered":
            st.warning(f"🛡️ **Safety Override Activated:** The model leaned towards 'Healthy', but failed to meet the safety margin. Displaying the most likely underlying infection.")
        elif res['confidence'] < 70.0:
            st.warning("⚠️ **Low Confidence Alert:** Displaying alternative possibilities.")

        st.markdown("---")
        st.markdown(f"#### 📋 Recommendations ({target_language})")
        
        disclaimer = "The Cloud API detects a broad range of diseases, but AI models can make mistakes." if "Cloud API" in res['source'] else "This local AI strictly identifies 6 common diseases. Unlisted diseases will cause inaccurate results."
        st.markdown(f'<div class="disclaimer-box"><b>⚠️ Safety Warning:</b> {disclaimer} <b>Always ensure a visual inspection by an expert before applying chemicals.</b></div>', unsafe_allow_html=True)

        # Apply translations (Cached to prevent repetitive API billing)
        symp_t = translate_text(res['symptoms'], target_language)
        cult_t = translate_text(res['cultural'], target_language)
        org_t  = translate_text(res['organic'], target_language)
        chem_t = translate_text(res['chemical'], target_language)

        st.info(f"**Symptoms:**\n{symp_t}\n\n**Cultural Control:**\n{cult_t}\n\n**Organic Control:**\n{org_t}\n\n**Chemical Control:**\n{chem_t}")

        # Display Top Alternative Predictions
        if len(res['top_preds']) > 0 and res['top_preds'][0][0] != "Safety Override Triggered":
            for i, (name, prob, raw_c) in enumerate(res['top_preds']):
                st.markdown(f"**#{i+1} Potential Match:** {name} ({prob:.1f}%)")

        st.markdown('</div>', unsafe_allow_html=True) 

        # Append structured block to report export
        report_lines.append(f"[IMAGE {idx+1} ANALYSIS]")
        report_lines.append(f"Identified   : {res['disease']} ({res['priority']})")
        report_lines.append(f"Confidence   : {res['confidence']:.1f}%")
        report_lines.append(f"\nRecommendations:\nSymptoms:\n{symp_t}\nCultural:\n{cult_t}\nOrganic:\n{org_t}\nChemical:\n{chem_t}")
        report_lines.append("\n----------------------------------------\n")

    # Generate the Downloadable File dynamically 
    full_report_text = "\n".join(report_lines)
    st.download_button(
        label="📥 Download Complete Field Report (.txt)",
        data=full_report_text,
        file_name=f"CocoaGuard_Report_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
