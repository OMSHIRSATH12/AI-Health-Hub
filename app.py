import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import warnings
import logging

# Suppress a common warning from sklearn for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
# Hide the verbose INFO messages from the Google API in the terminal
logging.getLogger('google.generativeai').setLevel(logging.WARNING)


# --- Page Configuration ---
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

# --- API Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("API Key not found or invalid. Please ensure you have a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
    st.stop()

# --- Gemini LLM Function ---
# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(question, chat_history):
    api_history = []
    for msg in chat_history:
        role = 'model' if msg['role'] == 'assistant' else msg['role']
        api_history.append({'role': role, 'parts': [{'text': msg['content']}]})

    chat = model.start_chat(history=api_history)
    
    prompt = f"""You are a helpful and knowledgeable AI Health Assistant. Your role is to provide clear, safe, and informative answers to health-related questions based on the provided chat history and the new question.
    When asked about a medical emergency, your first priority is to advise the user to contact local emergency services.
    Do not provide a diagnosis. Always recommend consulting a doctor.
    IMPORTANT RULE: If the user asks a question that is NOT related to health, wellness, or medicine, you MUST politely decline.
    Now, please answer the following question: {question}"""
    
    response_stream = chat.send_message(prompt, stream=True)
    
    for chunk in response_stream:
        yield chunk.text

# --- Machine Learning Model & Data Loading ---
@st.cache_resource
def train_symptom_checker_model():
    symptom_df = pd.read_csv("data/symptom_dataset_large.csv")
    if 'Unnamed: 133' in symptom_df.columns:
        symptom_df = symptom_df.drop('Unnamed: 133', axis=1)
    target = 'prognosis'
    features = symptom_df.columns.drop(target)
    X = symptom_df[features]
    y = symptom_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, features

# --- STREAMLIT USER INTERFACE ---
# --- Sidebar Content ---
with st.sidebar:
    st.title("🩺 AI Health Assistant")
    st.info("This project is a hackathon demo and not a substitute for professional medical advice.")

# --- Main Page Content ---
st.title("AI Health Assistant")

symptom_model, features = train_symptom_checker_model()
tab1, tab2, tab3 = st.tabs(["Super Chatbot", "Symptom Checker", "COVID-19 Dashboard"])

# --- Super Chatbot Tab (CORRECTED with st.form) ---
with tab1:
    st.header("Your Personal AI Health Advisor")
    st.write("Ask me anything about health. I can remember our conversation.")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        
    # Display the existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # --- NEW: Use a form for the chat input for more stability ---
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask your question:", placeholder="e.g., 'What are the first aid steps for a snake bite?'", key="chat_input")
        submitted = st.form_submit_button("Send")

    # Process the form submission
    if submitted and user_input:
        # Append and display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("The AI is thinking..."):
                response_stream = get_gemini_response(user_input, st.session_state.chat_history)
                full_response = st.write_stream(response_stream)
        
        # Append AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
        # Rerun the script to ensure the form is cleared and the new message is at the bottom
        st.rerun()

# --- Symptom Checker & Dashboard Tabs (No changes) ---
with tab2:
    # (Existing code, no changes)
    st.header("Advanced Symptom Checker")
    st.write("Select the symptoms you are experiencing from the list below.")
    with st.form("symptom_form"):
        user_symptoms = {}
        cols = st.columns(3)
        for i, feature in enumerate(features):
            user_symptoms[feature] = cols[i % 3].checkbox(feature.replace('_', ' ').title())
        submitted = st.form_submit_button("Predict Condition")
    if submitted:
        input_data = [int(user_symptoms[feature]) for feature in features]
        if sum(input_data) == 0:
            st.warning("Please select at least one symptom.")
        else:
            prediction = symptom_model.predict([input_data])
            st.subheader("Prediction:")
            st.success(f"Based on your symptoms, a possible condition could be: **{prediction[0]}**")
            st.warning("Disclaimer: This is an AI prediction based on a public dataset and is not a substitute for professional medical advice.")
with tab3:
    # (Existing code, no changes)
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