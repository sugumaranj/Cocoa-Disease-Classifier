# ==============================================================================
# FILE: app.py
# DESCRIPTION: Mobile-First Cocoa Disease Diagnosis Dashboard
#              Features: Dual-Engine AI, Cocoa Doctor Chatbot, Weather Alerts,
#                        Premium CSS Styling & Animations, Centered Custom Logo.
# ==============================================================================

import os
import socket
import datetime
import time 
import concurrent.futures 
import streamlit as st
import numpy as np
import requests  
from PIL import Image
from google import genai

# ==============================================================================
# 1. SETUP CLOUD API & LOCAL AI ENGINE
# ==============================================================================
# Try to load the local TensorFlow Lite model for offline edge processing
try:
    from ai_edge_litert.interpreter import Interpreter
    AI_ENGINE = "TFLite"
    TF_READY = True
except Exception as e:
    TF_READY = False
    TF_ERROR_MSG = str(e)

# Securely grab the Google API key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

# Securely grab the OpenWeather API key from Streamlit secrets
WEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")


# ==============================================================================
# 2. AGRONOMY DATABASES (KNOWLEDGE BASE)
# ==============================================================================
CLASS_NAMES = ['Anthracnose', 'CSSVD', 'Healthy', 'Monilia', 'Phytophthora', 'PodBorer']

SIMPLE_NAMES = {
    "Anthracnose": "Anthracnose (Fungal Spots)",
    "CSSVD": "Cocoa Swollen Shoot Virus (CSSVD)",
    "Healthy": "Healthy Crop",
    "Monilia": "Frosty Pod Rot",
    "Phytophthora": "Black Pod Rot",
    "PodBorer": "Cocoa Pod Borer (Pest)"
}

LOCAL_REMEDY_DB = {
    "Anthracnose": {
        "botanical": "Colletotrichum spp.",
        "priority": "HIGH",
        "symptoms": "- Sunken dark lesions on pods.\n- Pinkish spores appear in wet conditions.",
        "cultural": "- Prune shade trees to increase airflow.\n- Remove and burn infected pods immediately.",
        "organic": "- Apply Neem oil extract.\n- Use Trichoderma-based biological fungicides.",
        "chemical": "- Spray Copper Oxychloride (50% WP) @ 2.5g/L during rains."
    },
    "CSSVD": {
        "botanical": "Cocoa swollen shoot virus (Badnavirus)",
        "priority": "CRITICAL",
        "symptoms": "- Swollen stems and rounded, small pods.\n- Red vein banding on leaves.",
        "cultural": "- Uproot and burn infected trees immediately.\n- Plant barrier crops to stop mealybug spread.",
        "organic": "- Release natural predators (ladybugs) to control mealybug vectors.",
        "chemical": "- No chemical cure for the virus.\n- Use mild insecticides to control mealybug vectors if needed."
    },
    "Healthy": {
        "botanical": "Theobroma cacao (Healthy)",
        "priority": "NONE",
        "symptoms": "- Normal pod development.\n- No visible spots, rot, or lesions.",
        "cultural": "- Maintain proper field sanitation.\n- Ensure good soil drainage.",
        "organic": "- Apply organic compost to maintain soil microbiome.",
        "chemical": "- Apply standard NPK fertilizer.\n- No fungicidal intervention needed."
    },
    "Monilia": {
        "botanical": "Moniliophthora roreri",
        "priority": "HIGH",
        "symptoms": "- Premature ripening.\n- Thick, white/cream 'frosty' fungal spores on pod.",
        "cultural": "- Remove and bury mummified pods weekly.\n- Reduce overhead canopy shade.",
        "organic": "- Use biocontrol agents like Bacillus subtilis.",
        "chemical": "- Spray Copper Hydroxide (77% WP) @ 2g/L on developing pods."
    },
    "Phytophthora": {
        "botanical": "Phytophthora palmivora / P. megakarya",
        "priority": "CRITICAL",
        "symptoms": "- Hard, dark black spots.\n- Rot rapidly spreads to engulf the entire pod.",
        "cultural": "- Improve soil drainage trenches.\n- Harvest frequently; do not leave empty husks.",
        "organic": "- Apply Trichoderma viride enriched compost around the base.",
        "chemical": "- Spray Metalaxyl (8% WP) + Mancozeb (64% WP) @ 2.5g/L."
    },
    "PodBorer": {
        "botanical": "Conopomorpha cramerella",
        "priority": "HIGH",
        "symptoms": "- Premature patchy yellowing.\n- Tiny insect exit holes on the pod surface.",
        "cultural": "- Enclose young pods in biodegradable sleeves.\n- Harvest strictly every 7 days.",
        "organic": "- Use pheromone traps to disrupt mating.\n- Apply Neem-based sprays (Azadirachtin).",
        "chemical": "- Spray Cypermethrin (10% EC) @ 1ml/L safely away from harvest."
    }
}


# ==============================================================================
# 3. BACKGROUND UTILITY FUNCTIONS
# ==============================================================================
def check_internet_connection():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2)
        return True
    except OSError:
        pass
    return False

def fetch_gemini_diagnosis(prompt, img):
    response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, img])
    return response.text.strip()

def clean_api_text(text, unwanted_prefix):
    cleaned = text.replace(unwanted_prefix, "").replace(unwanted_prefix.lower(), "").strip()
    if not cleaned.startswith("-") and not cleaned.startswith("•"):
        cleaned = "- " + cleaned
    return cleaned.lstrip(":").strip()

@st.cache_data
def translate_text(text, target_lang):
    if target_lang == "English" or not text or client is None: 
        return text
    try:
        return client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"Translate this agricultural text perfectly into {target_lang}. Keep chemical names intact. Preserve all markdown formatting like bolding (**).\n\n{text}"
        ).text.strip()
    except:
        return text

@st.cache_data(ttl=3600) 
def fetch_local_weather(location_name, api_key):
    if not location_name or not api_key: 
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location_name}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

@st.cache_resource
def load_local_neural_network():
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
# 4. APP DESIGN & ✨ PREMIUM CUSTOM CSS ✨
# ==============================================================================
st.set_page_config(page_title="CocoaGuard 🌱", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
    /* PREMIUM BUTTON STYLING */
    .stButton>button {
        width: 100%; 
        border-radius: 14px; 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white !important;
        height: 3.8em; 
        font-weight: 800; 
        font-size: 1.15em; 
        letter-spacing: 0.5px;
        border: none; 
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.4); 
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #F57C00 0%, #E65100 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 152, 0, 0.6);
    }
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 10px rgba(255, 152, 0, 0.4);
    }
    
    /* FLOATING GLASSMORPHISM CARDS */
    .card {
        background-color: var(--secondary-background-color); 
        color: var(--text-color);
        padding: 25px; 
        border-radius: 18px; 
        border: 1px solid var(--border-color); 
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.12);
    }

    /* BADGES & ALERTS */
    .status-badge {
        padding: 8px 12px; border-radius: 20px; font-weight: 800; font-size: 0.85em;
        display: inline-block; margin-bottom: 12px; text-align: center; width: 100%;
        letter-spacing: 0.5px; text-transform: uppercase;
    }
    .api-badge {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .local-badge {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    
    .disclaimer-box {
        background-color: rgba(255, 193, 7, 0.12); color: var(--text-color); padding: 14px 18px; 
        border-radius: 10px; border-left: 6px solid #ffc107; margin-bottom: 15px; font-size: 0.9em;
        line-height: 1.5;
    }

    /* PRIORITY TEXT COLORS */
    .priority-CRITICAL { color: #d32f2f; font-weight: 900; background-color: #ffebee; padding: 4px 10px; border-radius: 6px;}
    .priority-HIGH { color: #f57c00; font-weight: 900; background-color: #fff3e0; padding: 4px 10px; border-radius: 6px;}
    .priority-LOW { color: #0097a7; font-weight: 900; background-color: #e0f7fa; padding: 4px 10px; border-radius: 6px;}
    .priority-NONE { color: #388e3c; font-weight: 900; background-color: #e8f5e9; padding: 4px 10px; border-radius: 6px;}
    .botanical-text { font-size: 0.9em; font-style: italic; color: #78909c; display: block; margin-top: 6px;}
    
    /* WEATHER DASHBOARD UPGRADES */
    .weather-box { 
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 30px; 
        border-radius: 18px; 
        border: none;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(33, 150, 243, 0.15);
        color: #0d47a1;
        text-align: center;
    }
    .weather-metric { font-size: 3em; font-weight: 900; color: #1565c0; text-shadow: 1px 1px 2px rgba(255,255,255,0.5); display: inline-block; margin: 10px 15px;}
    
    /* MODERN TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0px 0px;
        padding: 12px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 152, 0, 0.08);
        border-bottom-color: #FF9800 !important;
        border-bottom-width: 3px !important;
        color: #FF9800 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Session States memory
if "results" not in st.session_state: st.session_state.results = []
if "batch_analytics" not in st.session_state: st.session_state.batch_analytics = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []


# ==============================================================================
# 5. UI: SIDEBAR NAVIGATION & GLOBAL SETTINGS
# ==============================================================================
is_online = check_internet_connection() and client is not None

with st.sidebar:
    # 🌟 NEW LOGO IMPLEMENTATION 🌟
    # Uses columns to perfectly center the logo for both mobile and desktop screens
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            # Displays the logo file. Make sure your image is renamed to 'logo.png'
            st.image("logo.png", use_container_width=True)
        except Exception:
            # Fallback text just in case the image file hasn't been uploaded yet
            st.caption("(Upload logo.png)")
            
    # Clean spacing below the logo
    st.markdown("<h2 style='text-align: center; margin-top: -15px;'>CocoaGuard</h2>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("🌐 System Status")
    if is_online: st.markdown('<div class="status-badge api-badge">🟢 ONLINE (Cloud API)</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div class="status-badge local-badge">🔴 OFFLINE ({AI_ENGINE})</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    target_language = st.selectbox("🌍 Select Language:", ["English", "Tamil (தமிழ்)", "Malayalam (മലയാളം)", "Hindi (हिन्दी)", "Telugu (తెలుగు)"])
    
    st.markdown("---")
    st.header("📝 Farm Details")
    global_farm_name = st.text_input("Farm Owner Name")
    global_location = st.text_input("Farm Location (e.g., Coimbatore)")
    
    st.markdown("---")
    with st.expander("⚙️ Advanced AI Settings"):
        st.info("If the local AI confidence falls below this threshold, it overrides 'Healthy' to warn you of potential hidden threats.")
        safety_margin = st.slider("Safety Override Threshold (%)", min_value=70, max_value=99, value=90, step=1)


# ==============================================================================
# 6. UI: MAIN DASHBOARD & TABS
# ==============================================================================
st.title("🌱 CocoaGuard AI")
st.markdown("<h4 style='color: #888; font-weight: 400; margin-top: -15px; margin-bottom: 30px;'>Enterprise Agricultural Diagnostics Dashboard</h4>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Field Diagnostics", "👨‍⚕️ Cocoa Doctor", "🌤️ Weather & Alerts"])

# ------------------------------------------------------------------------------
# TAB 1: FIELD DIAGNOSTICS ENGINE
# ------------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Step 1: Capture or Upload Images")
    input_mode = st.radio("Select Input Method:", ["Local File Upload 📂", "Web Camera 📷"], horizontal=True, label_visibility="collapsed")

    uploaded_files = []
    if input_mode == "Web Camera 📷":
        camera_img = st.camera_input("Take a clear picture of the pod/leaf:")
        if camera_img: uploaded_files.append(camera_img)
    else:
        files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files: uploaded_files.extend(files)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        if st.button("🚀 RUN DIAGNOSTIC ENGINE", use_container_width=True):
            st.session_state.results = []
            st.session_state.batch_analytics = {"Healthy Crop": 0, "Anthracnose (Fungal Spots)": 0, "Cocoa Swollen Shoot Virus (CSSVD)": 0, "Frosty Pod Rot": 0, "Black Pod Rot": 0, "Cocoa Pod Borer (Pest)": 0}
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                raw_image = Image.open(file)
                api_success = False 
                diagnosis_source, final_disease_name, botanical_name, threat_priority = "", "", "", "NONE"
                final_confidence = 0.0
                symp, cult, org, chem = "", "", "", ""
                top_preds = []

                if is_online:
                    if idx > 0: time.sleep(2.5) 
                    with st.spinner("Analyzing via Cloud API..."):
                        prompt = (
                            "Analyze this cocoa crop image. Identify the disease (or state if Healthy). "
                            "Keep answers VERY SHORT. Use bullet points (-) for every new sentence. Do NOT mention universities. "
                            "Format EXACTLY using the pipe (|) character with 8 sections:\n"
                            "Disease Name|Botanical or Scientific Name|Confidence %|Priority Level (CRITICAL, HIGH, MODERATE, LOW, NONE)|Symptoms Bullets|Cultural Control Bullets|Organic Control Bullets|Chemical Control Bullets"
                        )
                        try:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(fetch_gemini_diagnosis, prompt, raw_image)
                                raw_text = future.result(timeout=60)
                            
                            if "|" in raw_text:
                                parts = raw_text.split("|")
                                if len(parts) >= 8:
                                    final_disease_name = parts[0].strip()
                                    botanical_name = parts[1].strip()
                                    try: final_confidence = float(parts[2].replace("%","").strip())
                                    except: final_confidence = 95.0 
                                    threat_priority = parts[3].strip().upper()
                                    symp = clean_api_text(parts[4], "Symptoms:")
                                    cult = clean_api_text(parts[5], "Cultural Control:")
                                    org = clean_api_text(parts[6], "Organic Control:")
                                    chem = clean_api_text(parts[7], "Chemical Control:")
                                    api_success = True
                                    diagnosis_source = "Cloud API 🟢"
                        except: 
                            api_success = False 

                if not api_success:
                    if local_model is None:
                        st.error("System Error: Local offline model unavailable.")
                        continue
                        
                    diagnosis_source = "Local AI Model 🔴"
                    with st.spinner("Processing via Local Edge AI..."):
                        img_resized = raw_image.resize((224, 224))
                        img_array = np.array(img_resized, dtype=np.float32)
                        img_batch = np.expand_dims(img_array, axis=0)
                        
                        input_details = local_model.get_input_details()
                        output_details = local_model.get_output_details()
                        local_model.set_tensor(input_details[0]['index'], img_batch)
                        local_model.invoke()
                        raw_predictions = local_model.get_tensor(output_details[0]['index'])[0]
                        
                        if np.max(raw_predictions) > 1.0: 
                            probs = (np.exp(raw_predictions - np.max(raw_predictions)) / np.exp(raw_predictions - np.max(raw_predictions)).sum()) * 100
                        else:
                            probs = raw_predictions * 100
                            
                        sorted_indices = np.argsort(probs)[::-1]
                        highest_index = sorted_indices[0]
                        raw_predicted_class = CLASS_NAMES[highest_index]
                        final_confidence = probs[highest_index]
                        
                        if raw_predicted_class == "Healthy" and final_confidence < safety_margin:
                            non_healthy_indices = [i for i in sorted_indices if CLASS_NAMES[i] != "Healthy"]
                            highest_index = non_healthy_indices[0]
                            raw_predicted_class = CLASS_NAMES[highest_index]
                            final_confidence = probs[highest_index]
                            top_preds.append(("Safety Override Triggered", 0.0, "N/A")) 

                        final_disease_name = SIMPLE_NAMES[raw_predicted_class]
                        db_entry = LOCAL_REMEDY_DB[raw_predicted_class]
                        botanical_name = db_entry["botanical"]
                        threat_priority = db_entry["priority"]
                        symp, cult, org, chem = db_entry['symptoms'], db_entry['cultural'], db_entry['organic'], db_entry['chemical']

                combo_text = f"**Symptoms:**\n{symp}\n\n**Cultural Control:**\n{cult}\n\n**Organic Control:**\n{org}\n\n**Chemical Control:**\n{chem}"

                st.session_state.results.append({
                    "image": raw_image, "disease": final_disease_name, "botanical": botanical_name,
                    "confidence": final_confidence, "priority": threat_priority, 
                    "combo_text": combo_text, "source": diagnosis_source, "top_preds": top_preds
                })

                if final_disease_name in st.session_state.batch_analytics:
                    st.session_state.batch_analytics[final_disease_name] += 1
                else: 
                    st.session_state.batch_analytics[final_disease_name] = 1
                    
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            st.success("✅ Analysis Complete!")

    # --- RENDER RESULTS ---
    if st.session_state.get("results"):
        report_lines = ["========================================", "   COCOAGUARD FIELD DIAGNOSIS REPORT", "========================================"]
        report_lines.append(f"Date/Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Farm Owner: {global_farm_name or 'Not Provided'}")
        report_lines.append(f"Location: {global_location or 'Not Provided'}\n\n[BATCH ANALYTICS SUMMARY]")
        for k, v in st.session_state.batch_analytics.items():
            if v > 0: report_lines.append(f"- {k}: {v} image(s)")
        report_lines.append("----------------------------------------\n")

        for idx, res in enumerate(st.session_state.results):
            st.markdown(f"### 🔍 Analysis: Image {idx+1}")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2: 
                # Display image with soft rounded corners
                st.image(res["image"], use_container_width=True)
            
            st.caption(f"Engine: {res['source']}")
            st.subheader(f"Identified: {res['disease']}")
            
            priority_class = f"priority-{res['priority']}" if res['priority'] in ["CRITICAL", "HIGH", "LOW", "NONE"] else "priority-HIGH"
            st.markdown(f'**Threat Priority:** <span class="{priority_class}">{res["priority"]}</span><br><span class="botanical-text">Scientific Name: {res["botanical"]}</span>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(int(res['confidence']) / 100.0)
            st.markdown(f"**Confidence Score:** {res['confidence']:.1f}%")

            if res['top_preds'] and res['top_preds'][0][0] == "Safety Override Triggered":
                st.warning("🛡️ **Safety Override Activated:** The model leaned towards 'Healthy', but failed to meet the safety margin. Displaying the most likely underlying infection.")

            st.markdown("---")
            st.markdown(f"#### 📋 Recommended Action Plan ({target_language})")
            st.markdown('<div class="disclaimer-box"><b>⚠️ Safety Warning:</b> Always ensure a visual inspection by an expert before applying chemicals.</div>', unsafe_allow_html=True)

            translated_combo = translate_text(res['combo_text'], target_language)
            st.info(translated_combo)

            st.markdown('</div>', unsafe_allow_html=True) 

            report_lines.append(f"[IMAGE {idx+1} ANALYSIS]")
            report_lines.append(f"Identified   : {res['disease']} ({res['priority']})")
            report_lines.append(f"Botanical    : {res['botanical']}")
            report_lines.append(f"Confidence   : {res['confidence']:.1f}%\n")
            report_lines.append(f"Recommendations:\n{translated_combo}\n")
            report_lines.append("\n----------------------------------------\n")

        full_report_text = '\ufeff' + "\n".join(report_lines)
        st.download_button(
            label="📥 Download Complete Field Report (.txt)",
            data=full_report_text.encode('utf-8'),
            file_name=f"CocoaGuard_Report_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )


# ------------------------------------------------------------------------------
# TAB 2: COCOA DOCTOR AI CHATBOT
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### 👨‍⚕️ Ask the Cocoa Doctor")
    st.caption(f"I am an expert agronomist. Ask me anything about cocoa farming, pests, or fertilizers! (I will reply in **{target_language}**)")

    chat_box = st.container(height=450)
    
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_prompt := st.chat_input("E.g., How do I prevent black pod rot during the rainy season?"):
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                if client is None:
                    st.error("⚠️ Cloud API is offline. The Cocoa Doctor requires an active internet connection.")
                else:
                    doc_prompt = (
                        f"You are a highly experienced Cocoa Crop Doctor helping a farmer. "
                        f"Provide a short, sweet, and highly accurate answer (maximum 3 sentences). "
                        f"Base your advice strictly on modern agricultural university research, but DO NOT mention any universities or research stations in your response. "
                        f"You MUST answer completely in the language: {target_language}.\n\n"
                        f"Farmer's Question: {user_prompt}"
                    )
                    try:
                        with st.spinner("Thinking..."):
                            chat_response = client.models.generate_content(model='gemini-2.5-flash', contents=doc_prompt)
                            st.markdown(chat_response.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": chat_response.text})
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "quota" in error_msg.lower():
                            st.error("⚠️ **API Busy:** The Cocoa Doctor is currently handling too many requests. Please wait 10 seconds and try again.")
                        else:
                            st.error(f"⚠️ **System Error:** {error_msg}")


# ------------------------------------------------------------------------------
# TAB 3: PROACTIVE WEATHER & FUNGAL ALERTS
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### 🌤️ Proactive Weather & Fungal Alerts")
    st.caption("Fungal diseases like Black Pod Rot explode during high humidity. Use this dashboard to predict local threats.")

    if not global_location:
        st.info("👈 **Action Required:** Please enter your **Farm Location** in the sidebar (e.g., 'Coimbatore') to view your local weather threat analysis.")
    elif not WEATHER_API_KEY:
        st.warning("⚠️ **System Error:** OpenWeather API key is missing. Please add it to your Streamlit secrets.")
    else:
        with st.spinner(f"Fetching real-time satellite data for {global_location}..."):
            weather_data = fetch_local_weather(global_location, WEATHER_API_KEY)
            
            if weather_data:
                humidity = weather_data['main']['humidity']
                temp = weather_data['main']['temp']
                condition = weather_data['weather'][0]['main'].lower()
                
                # Upgraded Premium Weather UI Box
                st.markdown(f'<div class="weather-box">'
                            f'<h4 style="margin:0; color:#1976d2;">Location: {weather_data["name"]}</h4>'
                            f'<span class="weather-metric">{temp}°C</span> | '
                            f'<span class="weather-metric">{humidity}% Humidity</span><br>'
                            f'<b style="font-size:1.2em; color:#0d47a1; text-transform:capitalize;">{condition}</b></div>', unsafe_allow_html=True)
                
                st.markdown("#### 🚨 Cocoa Crop Threat Analysis")
                
                if humidity >= 80 and ("rain" in condition or "drizzle" in condition or "thunderstorm" in condition):
                    st.error("⚠️ **CRITICAL FUNGAL THREAT:** High humidity and active rainfall detected. Preventive spraying for Black Pod Rot (Phytophthora) and Frosty Pod Rot is urgently recommended to prevent an outbreak.")
                elif humidity >= 80:
                    st.warning("⚠️ **Moderate Fungal Threat:** High ambient humidity detected. Monitor pods closely for Frosty Pod Rot and ensure shade trees are properly pruned to allow maximum airflow.")
                else:
                    st.success("✅ **Low Fungal Threat:** Weather conditions are currently stable. Standard maintenance and harvesting schedules can be followed.")
            else:
                st.error(f"⚠️ **Weather API Error:** Unable to retrieve data for '{global_location}'. If this is a brand-new OpenWeather API key, it can take 1-2 hours to activate globally. Otherwise, please check the city spelling.")
