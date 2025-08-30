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

# --- CONFIGURATIONS ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

# --- API & MODEL SETUP ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("API Key not found or invalid. Please check your .streamlit/secrets.toml file.")
    st.stop()

gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- DATA & ML MODEL FUNCTIONS ---
@st.cache_resource
def train_personalized_symptom_model():
    dataset_path = "data/personalized_symptoms.csv"
    if not os.path.exists(dataset_path):
        st.error("personalized_symptoms.csv not found.")
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

# --- HELPER FUNCTIONS (AI, VOICE) ---
def get_gemini_response(question, chat_history, language_name):
    api_history = []
    for msg in chat_history:
        role = 'model' if msg['role'] == 'assistant' else msg['role']
        api_history.append({'role': role, 'parts': [{'text': msg['content']}]})
        
    # --- THIS IS THE CORRECTED LINE ---
    chat = gemini_model.start_chat(history=api_history)
    
    prompt = f"""You are an AI Health Assistant. The user's preferred language is {language_name}. You MUST respond in this language. Your response will be converted to audio, so keep your answers conversational and reasonably concise.
    **CRITICAL INSTRUCTION: Analyze the entire CHAT HISTORY provided to understand the context of the conversation. The user may ask follow-up questions using pronouns like "it", "that", or "they". You must use the history to figure out what these pronouns refer to.**
    Your standard rules are:
    1. Provide safe, informative answers to health questions.
    2. Advise contacting emergency services for any medical emergency.
    3. Do not provide a diagnosis. Always recommend consulting a doctor.
    4. If the user asks an off-topic question, politely decline in their preferred language.
    Based on the full conversation history, provide a helpful and context-aware response to the user's latest message.
    Latest User Message: "{question}" """
    
    response_stream = chat.send_message(prompt, stream=True)
    for chunk in response_stream:
        yield chunk.text

@st.cache_data
def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")
        return None

def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio controls autoplay="true" style="display:none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# --- STREAMLIT UI ---
with st.sidebar:
    st.title("🩺 AI Health Assistant")
    st.info("A hackathon demo project. Not a substitute for professional medical advice.")
    language_options = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
    selected_language_name = st.selectbox("Select Language", options=list(language_options.keys()))
    selected_language_code = language_options[selected_language_name]

st.title("AI Health Assistant")
symptom_model, all_features, gender_encoder, all_symptoms = train_personalized_symptom_model()

tab1, tab2, tab3 = st.tabs(["Super Chatbot (Voice Enabled)", "Personalized Predictor", "COVID-19 Dashboard"])

with tab1:
    st.header("Your Personal AI Health Advisor")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = None
    if st.button("🎤 Ask with Voice"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                st.info("Processing...")
                user_input = r.recognize_google(audio, language=selected_language_code)
            except Exception as e:
                st.warning("Could not process audio. Please try again or type your question.")
    
    text_input = st.chat_input(f"Or type your question in {selected_language_name}...")
    if text_input:
        user_input = text_input

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        critical_keywords = ['chest pain', 'breathing difficulty', 'suicide', 'emergency', 'heart attack', 'severe bleeding', 'unconscious']
        is_emergency = any(keyword in user_input.lower() for keyword in critical_keywords)
        
        with st.chat_message("assistant"):
            if is_emergency:
                with open("assets/alert.mp3", "rb") as f:
                    audio_bytes = f.read()
                autoplay_audio(audio_bytes)
                emergency_text = "Emergency situation detected. Please contact your local emergency services immediately. In India, you can dial 112."
                st.error(emergency_text)
                st.session_state.chat_history.append({"role": "assistant", "content": emergency_text})
            else:
                with st.spinner("The AI is thinking..."):
                    response_stream = get_gemini_response(user_input, st.session_state.chat_history, selected_language_name)
                    full_response = st.write_stream(response_stream)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                audio_bytes = text_to_speech(full_response, selected_language_code)
                if audio_bytes:
                    autoplay_audio(audio_bytes)

with tab2:
    st.header("Personalized Symptom Checker")
    st.write("Enter your details and symptoms for a more personalized prediction.")
    with st.form("personalized_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Enter your Age", min_value=0, max_value=120, value=25)
        gender = col2.selectbox("Select your Gender", options=['Male', 'Female'])
        selected_symptoms = st.multiselect("Select your symptoms:", options=sorted(all_symptoms))
        submitted = st.form_submit_button("Predict Condition")
    if submitted:
        if not selected_symptoms: st.warning("Please select at least one symptom.")
        else:
            user_input_df = pd.DataFrame(columns=all_features)
            user_input_df.loc[0, 'Age'] = age
            user_input_df.loc[0, 'Gender'] = gender_encoder.transform([gender])[0]
            for symptom in all_symptoms:
                user_input_df.loc[0, symptom] = 1 if symptom in selected_symptoms else 0
            user_input_df = user_input_df.fillna(0)
            prediction = symptom_model.predict(user_input_df[all_features])
            prediction_proba = symptom_model.predict_proba(user_input_df[all_features])
            st.subheader("Prediction:")
            st.success(f"Based on your details, a possible condition could be: **{prediction[0]}**")
            st.info(f"Confidence: {prediction_proba.max()*100:.2f}%")
            st.warning("Disclaimer: This is an AI prediction and not a substitute for professional medical advice.")

with tab3:
    st.header("COVID-19 India Dashboard")
    covid_df = pd.read_csv("data/covid_india_snapshot.csv")
    metric_options = ['Total Cases', 'Active', 'Discharged', 'Deaths', 'Discharge Ratio (%)', 'Death Ratio (%)']
    covid_df['Discharge Ratio (%)'] = round((covid_df['Discharged'] / covid_df['Total Cases']) * 100, 2)
    covid_df['Death Ratio (%)'] = round((covid_df['Deaths'] / covid_df['Total Cases']) * 100, 2)
    selected_metric = st.selectbox("Select a metric to visualize:", options=metric_options)
    if selected_metric:
        df_sorted = covid_df.sort_values(by=selected_metric, ascending=False)
        fig = px.bar(df_sorted, x='State/UTs', y=selected_metric, title=f"State-wise Comparison of {selected_metric}")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(covid_df)