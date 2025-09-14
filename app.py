import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import plotly.express as px
import warnings
import logging
import numpy as np
import os
from io import BytesIO
import speech_recognition as sr
from gtts import gTTS
import base64
import requests
from urllib.parse import quote
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime

# --- CONFIGURATIONS & INITIAL SETUP ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
st.set_page_config(page_title="AI Health Hub", page_icon="🩺", layout="wide")

# --- FUNCTION TO LOAD CUSTOM CSS ---
def load_css(file_name):
    """Loads a CSS file and injects it into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply the custom CSS
if os.path.exists("style/style.css"):
    load_css("style/style.css")

# --- API & MODEL SETUP ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
except Exception as e:
    st.error("API Key not found or invalid. Please check your .streamlit/secrets.toml file.")
    st.stop()

# --- DATA & ML MODEL FUNCTIONS ---
@st.cache_resource
def train_personalized_symptom_model():
    """Trains the ML model on the personalized symptom dataset. Cached to run only once for performance."""
    dataset_path = "data/personalized_symptoms.csv"
    if not os.path.exists(dataset_path):
        st.error("personalized_symptoms.csv not found. Please ensure it's in the 'data' folder.")
        st.stop()
    df = pd.read_csv(dataset_path)
    symptom_cols = [col for col in df.columns if 'Symptom_' in col]
    df_melted = df.melt(id_vars=['Disease', 'Gender', 'Age'], value_vars=symptom_cols, value_name='Symptom').dropna().drop('variable', axis=1)
    mlb = MultiLabelBinarizer()
    symptom_encoded = mlb.fit_transform(df_melted.groupby(['Disease', 'Gender', 'Age'])['Symptom'].apply(list))
    symptom_encoded_df = pd.DataFrame(symptom_encoded, columns=mlb.classes_)
    patient_info_df = df_melted.groupby(['Disease', 'Gender', 'Age']).first().reset_index()
    final_df = pd.concat([patient_info_df[['Disease', 'Gender', 'Age']], symptom_encoded_df], axis=1)
    le = LabelEncoder()
    final_df['Gender'] = le.fit_transform(final_df['Gender'])
    y = final_df['Disease']
    X = final_df.drop('Disease', axis=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, X.columns, le, mlb.classes_

# --- HELPER FUNCTIONS ---
def get_gemini_response(question, chat_history, language_name):
    """Sends a prompt to the Gemini API and returns the response."""
    api_history = []
    for msg in chat_history[:-1]:
        role = 'model' if msg['role'] == 'assistant' else msg['role']
        api_history.append({'role': role, 'parts': [{'text': msg['content']}]})
    chat = gemini_model.start_chat(history=api_history)
    prompt = f"""You are a specialized AI Health Assistant. Your persona is helpful, knowledgeable, and safe. You have one strict rule: you ONLY answer questions related to health, medicine, wellness, symptoms, diseases, or first-aid.
    If the user's question is NOT about a health topic, your ONLY response MUST be this exact sentence in {language_name}: "I am an AI Health Assistant and can only answer health-related questions."
    If the user's question IS about a health topic, provide a clear, informative, and safe answer in {language_name}.
    - Use the provided CHAT HISTORY to understand the context of the conversation for follow-up questions.
    - Never give a medical diagnosis. Always recommend consulting a professional doctor.
    - For any mention of a medical emergency, prioritize advising the user to contact their local emergency services.
    CHAT HISTORY: {chat_history[:-1]}
    USER'S LATEST QUESTION: "{question}"
    """
    response = chat.send_message(prompt)
    return response.text

def get_recommendations(disease, language_name):
    """Generates medicine and diet recommendations for a given disease."""
    prompt = f"""A user has been predicted to have the condition: {disease}.
    Your task is to provide helpful, safe, and general recommendations for this condition in the user's preferred language ({language_name}).
    Structure your response with the following clear markdown sections:
    - **Common Medicines:** List a few common, over-the-counter medicines. Start with a clear disclaimer that the user MUST consult a doctor or pharmacist before taking any medication.
    - **Dietary Recommendations:** Suggest simple, helpful dietary choices.
    - **General Precautions:** Provide a list of general advice, like getting rest or when to see a doctor.
    Keep the language simple and easy to understand. This is for informational purposes only."""
    response = gemini_model.generate_content(prompt)
    return response.text

def get_report_summary(report_text, language_name):
    """Summarizes an uploaded medical report using the Gemini API."""
    prompt = f"""You are an expert medical analyst. Your task is to read the following medical report text and summarize it in simple, easy-to-understand language for a patient.
    Structure your summary with the following clear markdown sections:
    - **Key Findings:** A bulleted list of the most important results or observations.
    - **Doctor's Notes Summary:** A simple explanation of what the doctor's notes mean.
    - **Next Steps:** A bulleted list of recommended actions for the patient, if any are mentioned.
    IMPORTANT: This is not a diagnosis. Your summary should reflect only the information present in the report. Translate the final summary into {language_name}.
    MEDICAL REPORT TEXT: --- {report_text} ---
    """
    response = gemini_model.generate_content(prompt)
    return response.text

@st.cache_data
def fetch_health_news(api_key, search_term="", language_code='en'):
    """Fetches relevant health news from the NewsAPI using a smart query."""
    primary_keywords = ["healthcare", "medical", "clinical", "pharmaceutical"]
    secondary_keywords = ["disease", "treatment", "vaccine", "symptom", "hospital", "patient", "therapy", "diagnosis", "surgery", "neurosurgeon", "psychiatrist", "ayurveda"]
    base_query = f"({' OR '.join(primary_keywords)}) AND ({' OR '.join(secondary_keywords)})"
    final_query = base_query
    if search_term and search_term.strip():
        final_query += f" AND ({search_term})"
    encoded_query = quote(final_query)
    url = f"https://newsapi.org/v2/everything?q={encoded_query}&language={language_code}&sortBy=relevancy&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except requests.exceptions.RequestException:
        return []

@st.cache_data
def fetch_outbreak_news():
    """Fetches the latest disease outbreak news from the WHO's public API."""
    url = "https://www.who.int/api/news/diseaseoutbreaknews"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("value", [])
    except requests.exceptions.RequestException:
        return []

def autoplay_audio(audio_bytes: bytes):
    """Plays audio in the background without a visible player."""
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'<audio controls autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(md, unsafe_allow_html=True)

@st.cache_data
def text_to_speech(text: str, lang_code: str):
    """Converts a string of text to speech audio bytes."""
    try:
        if not isinstance(text, str) or not text.strip(): return None
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except: return None

@st.cache_data
def get_location_coords(city_name):
    """Gets latitude and longitude for a city name."""
    geolocator = Nominatim(user_agent="ai_health_hub")
    try:
        location = geolocator.geocode(city_name)
        if location: return (location.latitude, location.longitude)
    except: return None
    return None

@st.cache_data
def get_nearby_hospitals(lat, lon):
    """Gets a list of nearby hospitals from OpenStreetMap."""
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""[out:json];(node["amenity"="hospital"](around:5000,{lat},{lon});way["amenity"="hospital"](around:5000,{lat},{lon});relation["amenity"="hospital"](around:5000,{lat},{lon}););out center;"""
    try:
        response = requests.get(overpass_url, params={'data': overpass_query})
        response.raise_for_status()
        data = response.json()
        hospitals = []
        for element in data['elements']:
            if 'lat' in element and 'lon' in element:
                hospitals.append({'name': element.get('tags', {}).get('name', 'Hospital'),'lat': element['lat'],'lon': element['lon']})
        return hospitals
    except: return []

# --- STREAMLIT UI ---
# Sidebar for global controls.
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", use_column_width=True)
    st.title("AI Health Hub")
    st.info("A hackathon demo project.")
    language_options = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
    selected_language_name = st.selectbox("Select Language", options=list(language_options.keys()))
    selected_language_code = language_options[selected_language_name]

st.title("AI Health Hub")

# Load the ML model and its data.
symptom_model, all_features, gender_encoder, all_symptoms = train_personalized_symptom_model()

# Create the main UI tabs.
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Super Chatbot", "Report Summarizer", "Personalized Predictor", "COVID-19 Dashboard", "Health News", "Health Hub"])

# --- TAB 1: SUPER CHATBOT ---
with tab1:
    st.header("Your Personal AI Health Advisor")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history, including the map if an emergency was triggered
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("is_emergency"):
                with st.spinner("Loading map..."):
                    coords = get_location_coords(message["location"])
                    if coords:
                        hospitals = get_nearby_hospitals(coords[0], coords[1])
                        m = folium.Map(location=coords, zoom_start=13)
                        folium.Marker(coords, popup="Your Approx. Location", icon=folium.Icon(color='red')).add_to(m)
                        for hospital in hospitals:
                            folium.Marker([hospital['lat'], hospital['lon']], popup=hospital['name'], icon=folium.Icon(color='blue', icon='hospital-o', prefix='fa')).add_to(m)
                        st_folium(m, width=700, height=400)
                    else:
                        st.warning("Could not find location to display map.")
    
    # This logic now handles the two-step emergency state
    if st.session_state.get("emergency_detected"):
        with st.chat_message("assistant"):
            with open("assets/alert.mp3", "rb") as f: audio_bytes = f.read()
            autoplay_audio(audio_bytes)
            st.error("Emergency detected. Please enter your city to find nearby hospitals.")
            
            with st.form("location_form"):
                location_input = st.text_input("Enter your city (e.g., Nashik, India)", value="Nashik")
                submitted = st.form_submit_button("Find Hospitals")

            if submitted and location_input:
                st.session_state.emergency_detected = False
                st.session_state.chat_history.append({"role": "assistant", "content": f"Showing hospitals near {location_input}", "is_emergency": True, "location": location_input})
                st.rerun()
    else:
        # This function handles a single, stable conversation turn
        def handle_chat_turn(prompt):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            lower_prompt = prompt.lower()
            critical_keywords = ['chest pain', 'breathing difficulty', 'suicide', 'emergency', 'heart attack', 'severe bleeding', 'unconscious']
            info_keywords = ['what', 'symptoms', 'tell me', 'about', 'information', 'define', 'explain', 'is a']
            is_emergency = any(keyword in lower_prompt for keyword in critical_keywords) and not any(keyword in lower_prompt for keyword in info_keywords)
            
            if is_emergency:
                st.session_state.emergency_detected = True
            else:
                with st.spinner("The AI is thinking..."):
                    full_response = get_gemini_response(prompt, st.session_state.chat_history, selected_language_name)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    audio_bytes = text_to_speech(full_response, selected_language_code)
                    if audio_bytes: autoplay_audio(audio_bytes)
            st.rerun()

        # Voice input button
        if st.button("🎤 Ask with Voice"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("Listening...")
                try:
                    audio = r.listen(source, timeout=5)
                    voice_prompt = r.recognize_google(audio, language=selected_language_code)
                    handle_chat_turn(voice_prompt)
                except:
                    st.warning("Could not process audio.")
        
        # Text input
        if text_prompt := st.chat_input(f"Or type your question..."):
            handle_chat_turn(text_prompt)

# --- TAB 2: REPORT SUMMARIZER ---
with tab2:
    st.header("AI Medical Report Summarizer")
    st.write("Upload a text (.txt) file of your medical report, and the AI will provide a simple summary.")
    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
    if uploaded_file is not None:
        report_text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("Original Report Text", report_text, height=250)
        if st.button("Summarize Report"):
            with st.spinner("The AI is analyzing your report..."):
                summary = get_report_summary(report_text, selected_language_name)
                st.subheader("Your Report Summary")
                st.markdown(summary)

# --- TAB 3: PERSONALIZED PREDICTOR ---
with tab3:
    st.header("Personalized Symptom Checker")
    st.write("Enter your details and symptoms for a more personalized prediction.")
    with st.form("personalized_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Enter your Age", 0, 120, 25)
        gender = col2.selectbox("Select your Gender", options=gender_encoder.classes_)
        selected_symptoms = st.multiselect("Select your symptoms:", options=sorted(all_symptoms))
        submitted = st.form_submit_button("Predict Condition")
    if submitted:
        if not selected_symptoms: st.warning("Please select at least one symptom.")
        else:
            user_input_df = pd.DataFrame(columns=all_features)
            user_input_df.loc[0, 'Age'] = age
            user_input_df.loc[0, 'Gender'] = gender_encoder.transform([gender])[0]
            for symptom in all_symptoms: user_input_df.loc[0, symptom] = 1 if symptom in selected_symptoms else 0
            user_input_df = user_input_df.fillna(0)[all_features]
            prediction = symptom_model.predict(user_input_df)
            prediction_proba = symptom_model.predict_proba(user_input_df)
            st.subheader("Prediction Result:")
            st.success(f"Based on your details, a possible condition could be: **{prediction[0]}**")
            st.info(f"Confidence: {prediction_proba.max()*100:.2f}%")
            st.warning("Disclaimer: This is not professional medical advice.")
            with st.spinner("Generating recommendations..."):
                recommendations = get_recommendations(prediction[0], selected_language_name)
                st.subheader("Recommendations")
                st.markdown(recommendations)

# --- TAB 4: COVID-19 DASHBOARD ---
with tab4:
    st.header("COVID-19 India Dashboard")
    covid_df = pd.read_csv("data/covid_india_snapshot.csv")
    metric_options = ['Total Cases', 'Active', 'Discharged', 'Deaths', 'Discharge Ratio (%)', 'Death Ratio (%)']
    covid_df['Discharge Ratio (%)'] = round((covid_df['Discharged'] / covid_df['Total Cases']) * 100, 2)
    covid_df['Death Ratio (%)'] = round((covid_df['Deaths'] / covid_df['Total Cases']) * 100, 2)
    selected_metric = st.selectbox("Select a metric to visualize:", options=metric_options)
    if selected_metric:
        df_sorted = covid_df.sort_values(by=selected_metric, ascending=False)
        fig = px.bar(df_sorted, x='State/UTs', y=selected_metric, title=f"State-wise Comparison of {selected_metric}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(covid_df)

# --- TAB 5: HEALTH NEWS ---
with tab5:
    st.header(f"Latest Health News")
    with st.form("news_search_form"):
        search_query = st.text_input("Search for a specific topic (e.g., 'diabetes')", placeholder="Leave blank for general health news")
        submitted = st.form_submit_button("Search News")
    articles = fetch_health_news(NEWS_API_KEY, search_term=search_query, language_code=selected_language_code)
    if articles:
        for article in articles[:10]:
            st.subheader(article['title'])
            if article.get('urlToImage'): st.image(article['urlToImage'])
            st.write(article.get('description', 'No description available.'))
            st.markdown(f"<a href='{article['url']}' target='_blank' rel='noopener noreferrer'>Read Full Article</a>", unsafe_allow_html=True)
            st.write(f"**Source:** {article['source']['name']}")
            st.divider()
    elif submitted: st.warning("No news articles found for your specific search.")
    else: st.info("Showing general health news. Use the search bar above to find a specific topic.")

# --- TAB 6: HEALTH HUB ---
with tab6:
    st.header("Health Hub: Alerts, Myths & FAQs")
    st.subheader("Live WHO Disease Outbreak Alerts")
    outbreaks = fetch_outbreak_news()
    if outbreaks:
        for alert in outbreaks[:5]:
            with st.expander(f"**{alert.get('Title', 'No Title')}** - Published on {datetime.fromisoformat(alert.get('PublicationDate', '').replace('Z', '+00:00')).strftime('%d %B %Y')}"):
                st.markdown(f"<a href='{alert.get('ItemDefaultUrl', '#')}' target='_blank' rel='noopener noreferrer'>Read the full alert on the WHO website</a>", unsafe_allow_html=True)
    else: st.info("Could not retrieve outbreak news from WHO at this time.")
    st.divider()
    st.subheader("Health Myths vs. Facts")
    with st.expander("Myth: Cracking your knuckles causes arthritis."):
        st.write("**Fact:** While it might be an annoying habit, there is no scientific evidence to suggest that cracking your knuckles leads to arthritis.")
    with st.expander("Myth: You need to drink 8 glasses of water a day."):
        st.write("**Fact:** This is a general guideline, not a strict medical requirement. Your hydration needs depend on your activity level, climate, and overall health.")
    st.divider()
    st.subheader("Frequently Asked Questions (FAQs)")
    with st.expander("What should I do for a common cold?"):
        st.write("Focus on rest and hydration. See a doctor if symptoms worsen or last more than 10 days.")
    with st.expander("How can I lower my blood pressure naturally?"):
        st.write("Lifestyle changes like regular exercise, a healthy diet, and reducing stress can help. Always discuss with your doctor.")

