# ==============================================================================
# FILE: app.py
# DESCRIPTION: Mobile-First Cocoa Disease Diagnosis Dashboard & AI Chatbot
#              Features: Interactive Treatment Calculator, AI Chatbot (Cocoa Doctor),
#                        Dual-Engine Architecture (Cloud API + Edge TFLite Fallback),
#                        Dynamic Translation Caching, & UTF-8 Report Exports.
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
# 1. SETUP CLOUD API & LOCAL AI ENGINE
# ==============================================================================
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

# ==============================================================================
# 2. DATABASES & CALCULATOR SETTINGS
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

DOSAGE_MAP = {
    "Anthracnose": {"chem": "Copper Oxychloride (50% WP)", "rate": 2.5, "unit": "g"},
    "Monilia": {"chem": "Copper Hydroxide (77% WP)", "rate": 2.0, "unit": "g"},
    "Phytophthora": {"chem": "Metalaxyl (8% WP) + Mancozeb (64% WP)", "rate": 2.5, "unit": "g"},
    "PodBorer": {"chem": "Cypermethrin (10% EC)", "rate": 1.0, "unit": "ml"}
}

# Offline database (Updated with Botanical names for the UI)
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
# 3. BACKGROUND FUNCTIONS
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
    """Caches translations. We pass the whole text block (including headers) so they get translated too!"""
    if target_lang == "English" or not text or client is None: 
        return text
    try:
        return client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"Translate this agricultural text perfectly into {target_lang}. Keep chemical names intact. Preserve all markdown formatting like bolding (**).\n\n{text}"
        ).text.strip()
    except:
        return text

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
# 4. APP DESIGN & CSS
# ==============================================================================
st.set_page_config(page_title="CocoaGuard 🌱", page_icon="🌱", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 12px; background-color: #ff9800; color: white;
        height: 3.5em; font-weight: 900; font-size: 1.1em; border: 2px solid #f57c00; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #e65100; transform: scale(1.02);}
    .status-badge {
        padding: 6px 10px; border-radius: 15px; font-weight: bold; font-size: 0.8em;
        display: inline-block; margin-bottom: 10px; text-align: center; width: 100%;
    }
    .api-badge {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .local-badge {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .card {
        background-color: var(--secondary-background-color); color: var(--text-color);
        padding: 20px; border-radius: 12px; border: 1px solid var(--border-color); margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: rgba(255, 193, 7, 0.15); color: var(--text-color); padding: 12px; 
        border-radius: 8px; border-left: 5px solid #ffc107; margin-bottom: 15px; font-size: 0.9em;
    }
    .priority-CRITICAL { color: #d32f2f; font-weight: 900; background-color: #f8d7da; padding: 4px 8px; border-radius: 4px;}
    .priority-HIGH { color: #ff9800; font-weight: 900; background-color: rgba(255, 152, 0, 0.2); padding: 4px 8px; border-radius: 4px;}
    .priority-LOW { color: #0c5460; font-weight: 900; background-color: #d1ecf1; padding: 4px 8px; border-radius: 4px;}
    .priority-NONE { color: #155724; font-weight: 900; background-color: #d4edda; padding: 4px 8px; border-radius: 4px;}
    .botanical-text { font-size: 0.85em; font-style: italic; color: #888; display: block; margin-top: 4px;}
    .calc-box { 
        background-color: rgba(46, 125, 50, 0.1); padding: 15px; border-radius: 10px; 
        border: 1px solid #2e7d32; margin-top: 10px; color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

# Session States
if "results" not in st.session_state: st.session_state.results = []
if "batch_analytics" not in st.session_state: st.session_state.batch_analytics = {}
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ==============================================================================
# 5. UI: SIDEBAR
# ==============================================================================
is_online = check_internet_connection() and client is not None

with st.sidebar:
    st.header("🌐 System Status")
    if is_online: st.markdown('<div class="status-badge api-badge">🟢 ONLINE (Cloud API)</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div class="status-badge local-badge">🔴 OFFLINE ({AI_ENGINE})</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    target_language = st.selectbox("🌍 Select Language:", ["English", "Tamil (தமிழ்)", "Malayalam (മലയാളം)", "Hindi (हिन्दी)", "Telugu (తెలుగు)"])
    
    st.markdown("---")
    st.header("📝 Farm Details")
    global_farm_name = st.text_input("Farm Owner Name")
    global_location = st.text_input("Farm Location")
    
    st.markdown("---")
    with st.expander("⚙️ Advanced AI Settings"):
        safety_margin = st.slider("Safety Override Threshold (%)", min_value=70, max_value=99, value=90, step=1)

# ==============================================================================
# 6. UI: MAIN INTERFACE (TABS)
# ==============================================================================
st.title("🌱 CocoaGuard")
st.markdown("### Enterprise Agricultural Dashboard")

tab1, tab2 = st.tabs(["📸 Field Diagnostics", "👨‍⚕️ Cocoa Doctor Chatbot"])

# ------------------------------------------------------------------------------
# TAB 1: FIELD DIAGNOSTICS & CALCULATOR
# ------------------------------------------------------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_mode = st.radio("Select Input Method:", ["Local File Upload / Phone Camera 📂", "Built-in Web Camera 📷"], horizontal=True)

    uploaded_files = []
    if input_mode == "Built-in Web Camera 📷":
        camera_img = st.camera_input("Take a clear picture of the pod/leaf:")
        if camera_img: uploaded_files.append(camera_img)
    else:
        files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files: uploaded_files.extend(files)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        if st.button("🚀 RUN DIAGNOSTIC", use_container_width=True):
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

                # CLOUD API ENGINE
                if is_online:
                    if idx > 0: time.sleep(2.5) 
                    with st.spinner("Analyzing via Cloud API..."):
                        # Prompt updated to extract Botanical Name strictly
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
                        except: api_success = False

                # EDGE TFLITE ENGINE FALLBACK
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
                        
                        # SAFETY OVERRIDE
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

                        if final_confidence < 70.0 and len(top_preds) == 0:
                            for i in range(min(2, len(sorted_indices))):
                                c_name = CLASS_NAMES[sorted_indices[i]]
                                top_preds.append((SIMPLE_NAMES[c_name], probs[sorted_indices[i]], c_name))

                # Combine text block so headers get translated properly
                combo_text = f"**Symptoms:**\n{symp}\n\n**Cultural Control:**\n{cult}\n\n**Organic Control:**\n{org}\n\n**Chemical Control:**\n{chem}"

                st.session_state.results.append({
                    "image": raw_image, "disease": final_disease_name, "botanical": botanical_name,
                    "confidence": final_confidence, "priority": threat_priority, 
                    "combo_text": combo_text, "source": diagnosis_source, "top_preds": top_preds
                })

                if final_disease_name in st.session_state.batch_analytics:
                    st.session_state.batch_analytics[final_disease_name] += 1
                else: st.session_state.batch_analytics[final_disease_name] = 1
                    
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
            st.success("✅ Analysis Complete!")

    # RENDER DIAGNOSTIC RESULTS
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
            with col_img2: st.image(res["image"], use_container_width=True)
            
            st.markdown(f"**Data Source:** {res['source']}")
            st.subheader(f"Identified: {res['disease']}")
            
            # Botanical Name injected below Priority
            priority_class = f"priority-{res['priority']}" if res['priority'] in ["CRITICAL", "HIGH", "LOW", "NONE"] else "priority-HIGH"
            st.markdown(f'**Threat Priority:** <span class="{priority_class}">{res["priority"]}</span><br><span class="botanical-text">Botanical Name: {res["botanical"]}</span>', unsafe_allow_html=True)
            
            st.progress(int(res['confidence']) / 100.0)
            st.markdown(f"**Confidence Score:** {res['confidence']:.1f}%")

            if res['top_preds'] and res['top_preds'][0][0] == "Safety Override Triggered":
                st.warning("🛡️ **Safety Override Activated:** The model leaned towards 'Healthy', but failed to meet the safety margin. Displaying the most likely underlying infection.")

            st.markdown("---")
            st.markdown(f"#### 📋 Recommendations ({target_language})")
            st.markdown('<div class="disclaimer-box"><b>⚠️ Safety Warning:</b> Always ensure a visual inspection by an expert before applying chemicals.</div>', unsafe_allow_html=True)

            # Translate the entire block (which translates headers automatically)
            translated_combo = translate_text(res['combo_text'], target_language)
            st.info(translated_combo)

            calc_report_text = ""
            
            # INTERACTIVE CALCULATOR
            dis_name_lower = res["disease"].lower()
            calc_data = None
            if "anthracnose" in dis_name_lower: calc_data = DOSAGE_MAP["Anthracnose"]
            elif "frosty" in dis_name_lower or "monilia" in dis_name_lower: calc_data = DOSAGE_MAP["Monilia"]
            elif "black pod" in dis_name_lower or "phytophthora" in dis_name_lower: calc_data = DOSAGE_MAP["Phytophthora"]
            elif "borer" in dis_name_lower or "pest" in dis_name_lower: calc_data = DOSAGE_MAP["PodBorer"]
            
            if calc_data:
                st.markdown("#### 🧮 Treatment Calculator")
                col_c1, col_c2 = st.columns(2)
                with col_c1: acres = st.number_input("Spray Area (Acres)", min_value=0.1, value=1.0, step=0.5, key=f"acre_{idx}")
                with col_c2: water = st.number_input("Water Volume (Liters/Acre)", min_value=50, value=200, step=10, key=f"water_{idx}")
                
                total_water = acres * water
                total_chem = total_water * calc_data["rate"]
                chem_display = f"{total_chem/1000:,.2f} kg" if calc_data["unit"] == "g" and total_chem >= 1000 else f"{total_chem/1000:,.2f} Liters" if calc_data["unit"] == "ml" and total_chem >= 1000 else f"{total_chem:,.1f} {calc_data['unit']}"
                
                st.markdown(f'<div class="calc-box">🧪 <b>Required Mix:</b> Add <b style="color:#4caf50; font-size:1.2em;">{chem_display}</b> of {calc_data["chem"]} to <b style="color:#29b6f6; font-size:1.2em;">{total_water:,.0f} Liters</b> of water.</div>', unsafe_allow_html=True)
                calc_report_text = f"\n[Treatment Calculation for {acres} Acres]\n- Water Needed: {total_water:,.0f} Liters\n- Chemical Needed: {chem_display} of {calc_data['chem']}"

            st.markdown('</div>', unsafe_allow_html=True) 

            # Add to text file report
            report_lines.append(f"[IMAGE {idx+1} ANALYSIS]")
            report_lines.append(f"Identified   : {res['disease']} ({res['priority']})")
            report_lines.append(f"Botanical    : {res['botanical']}")
            report_lines.append(f"Confidence   : {res['confidence']:.1f}%\n")
            report_lines.append(f"Recommendations:\n{translated_combo}\n")
            if calc_report_text: report_lines.append(calc_report_text)
            report_lines.append("\n----------------------------------------\n")

        # 📥 FIX: UTF-8 BOM encoding so Windows Notepad reads regional languages perfectly
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

    # Render previous chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box
    if user_prompt := st.chat_input("E.g., How do I prevent black pod rot during the rainy season?"):
        # Add user question to UI
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # AI Response Generation
        with st.chat_message("assistant"):
            if client is None:
                st.error("⚠️ Cloud API is offline. The Cocoa Doctor requires an active internet connection.")
            else:
                # Prompt Engineering to enforce rules (short answers, invisible university source, correct language)
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
                    # Enterprise-grade error handling to check for API rate limits
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("⚠️ **API Busy:** The Cocoa Doctor is currently handling too many requests. Please wait 10 seconds and try again.")
                    else:
                        st.error(f"⚠️ **System Error:** {error_msg}")
