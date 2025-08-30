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
from gtts import gTTS
import base64

# --- CONFIGURATIONS ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")

# --- API & MODEL SETUP ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("API Key not found or invalid. Please check your .streamlit/secrets.toml file.")
    st.stop()

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

# --- HELPER FUNCTIONS ---
def get_gemini_response(question, chat_history, language_name):
    api_history = []
    for msg in chat_history[:-1]:
        role = 'model' if msg['role'] == 'assistant' else msg['role']
        api_history.append({'role': role, 'parts': [{'text': msg['content']}]})
    chat = gemini_model.start_chat(history=api_history)
    prompt = f"""You are a specialized AI Health Assistant. Your ONLY function is to answer health-related questions. You must adhere to the following rules strictly.

    CHAT HISTORY:
    {chat_history[:-1]}

    USER'S LATEST QUESTION: "{question}"

    YOUR TASK:
    1.  First, analyze the user's latest question. Is it related to health, medicine, wellness, symptoms, diseases, or first-aid?
    2.  IF THE QUESTION IS NOT a health question, you must respond with ONLY this exact sentence in the user's preferred language ({language_name}): "I am an AI Health Assistant and can only answer health-related questions." DO NOT answer the off-topic question.
    3.  IF THE QUESTION IS a health question, then provide a helpful, safe, and informative answer in {language_name}. Use the chat history to understand context for follow-up questions. Never provide a diagnosis. Always recommend consulting a doctor. For emergencies, prioritize advising the user to contact emergency services.
    """
    response = chat.send_message(prompt)
    return response.text

def autoplay_audio(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    md = f'<audio controls autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(md, unsafe_allow_html=True)

@st.cache_data
def text_to_speech(text: str, lang_code: str):
    try:
        if not isinstance(text, str) or not text.strip():
            return None
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.warning(f"Could not process text-to-speech: {e}")
        return None

# --- STREAMLIT UI ---
with st.sidebar:
    st.title("🩺 AI Health Assistant")
    st.info("A hackathon demo project.")
    language_options = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
    selected_language_name = st.selectbox("Select Language", options=list(language_options.keys()))
    selected_language_code = language_options[selected_language_name]
    st.toggle("Mute AI Voice", key="mute_voice", value=False)

st.title("AI Health Assistant")
symptom_model, all_features, gender_encoder, all_symptoms = train_personalized_symptom_model()

tab1, tab2, tab3 = st.tabs(["Super Chatbot", "Personalized Predictor", "COVID-19 Dashboard"])

with tab1:
    st.header("Your Personal AI Health Advisor")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a health question in {selected_language_name}..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            lower_prompt = prompt.lower()
            critical_keywords = ['chest pain', 'breathing difficulty', 'suicide', 'emergency', 'heart attack', 'severe bleeding', 'unconscious']
            info_keywords = ['what', 'symptoms', 'tell me', 'about', 'information', 'define', 'explain', 'is a']
            
            is_emergency_keyword_present = any(keyword in lower_prompt for keyword in critical_keywords)
            is_info_query = any(keyword in lower_prompt for keyword in info_keywords)
            is_true_emergency = is_emergency_keyword_present and not is_info_query

            if is_true_emergency:
                # The emergency beep now plays ALWAYS, regardless of the mute toggle.
                with open("assets/alert.mp3", "rb") as f:
                    audio_bytes = f.read()
                autoplay_audio(audio_bytes)
                
                emergency_text = "Emergency detected. Please contact local emergency services immediately (e.g., dial 112 in India)."
                st.error(emergency_text)
                st.session_state.chat_history.append({"role": "assistant", "content": emergency_text})
            else:
                with st.spinner("The AI is thinking..."):
                    full_response = get_gemini_response(prompt, st.session_state.chat_history, selected_language_name)
                    st.markdown(full_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                    # The conversational voice correctly respects the mute toggle.
                    if not st.session_state.mute_voice:
                        audio_bytes = text_to_speech(full_response, selected_language_code)
                        if audio_bytes:
                            autoplay_audio(audio_bytes)

with tab2:
    st.header("Personalized Symptom Checker")
    st.write("Enter your details and symptoms for a more personalized prediction.")
    with st.form("personalized_form"):
        col1, col2 = st.columns(2)
        age = col1.number_input("Enter your Age", 0, 120, 25)
        gender = col2.selectbox("Select your Gender", options=gender_encoder.classes_)
        selected_symptoms = st.multiselect("Select your symptoms:", options=sorted(all_symptoms))
        submitted = st.form_submit_button("Predict Condition")
    
    if submitted:
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            user_input_df = pd.DataFrame(columns=all_features)
            user_input_df.loc[0, 'Age'] = age
            user_input_df.loc[0, 'Gender'] = gender_encoder.transform([gender])[0]
            for symptom in all_symptoms:
                user_input_df.loc[0, symptom] = 1 if symptom in selected_symptoms else 0
            user_input_df = user_input_df.fillna(0)[all_features]
            prediction = symptom_model.predict(user_input_df)
            prediction_proba = symptom_model.predict_proba(user_input_df)
            
            st.subheader("Prediction Result:")
            st.success(f"Based on your details, a possible condition could be: **{prediction[0]}**")
            st.info(f"Confidence: {prediction_proba.max()*100:.2f}%")
            st.warning("Disclaimer: This is not professional medical advice.")

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