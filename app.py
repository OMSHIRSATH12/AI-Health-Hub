import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

# --- API Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("API Key not found or invalid. Please ensure you have a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
    st.stop()

# --- Gemini LLM Function (with the new, improved prompt) ---
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""You are a helpful and knowledgeable AI Health Assistant. 
    Your role is to provide clear, safe, and informative answers to health-related questions.
    When asked about a medical emergency (like snake bite, heart attack, severe bleeding), your first priority is to advise the user to contact local emergency services immediately.
    Provide first-aid steps only as something to do while waiting for professional help.
    Do not provide a diagnosis. Always recommend consulting a doctor for any medical advice.

    IMPORTANT RULE: If the user asks a question that is NOT related to health, wellness, first-aid, or medicine, you MUST politely decline to answer. You should state that you are an AI Health Assistant and can only answer health-related questions. Do not answer the off-topic question.
    
    Now, please answer the following question: {question}"""
    
    response = model.generate_content(prompt)
    return response.text

# --- Machine Learning Model & Data Loading ---
@st.cache_resource
def train_symptom_checker_model():
    symptom_df = pd.read_csv("data/symptom_data.csv")
    features = ['fever', 'cough', 'headache', 'sore_throat', 'fatigue', 'body_aches']
    target = 'disease'
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

# --- Super Chatbot Tab (Powered by Gemini) ---
with tab1:
    st.header("Your Personal AI Health Advisor")
    st.write("Ask me anything about health, from first-aid for a snake bite to questions about nutrition.")
    
    user_input = st.text_input("", placeholder="e.g., 'What are the first aid steps for a snake bite?'", key="chatbot_input")

    if st.button("Get Answer"):
        if user_input:
            with st.spinner("The AI is thinking..."):
                response = get_gemini_response(user_input)
                st.markdown(response)
        else:
            st.warning("Please enter a question.")

# --- Symptom Checker & Dashboard Tabs (No changes) ---
with tab2:
    st.header("Symptom Checker")
    st.write("Select your symptoms, and the AI will predict a possible condition.")
    user_symptoms = {}
    for feature in features:
        user_symptoms[feature] = st.checkbox(feature.replace('_', ' ').title())
    if st.button("Predict Condition"):
        input_data = [user_symptoms[feature] for feature in features]
        prediction = symptom_model.predict([input_data])
        st.subheader("Prediction:")
        st.success(f"Based on your symptoms, the model predicts you might have: **{prediction[0]}**")
        st.warning("Disclaimer: This is an AI prediction and not a substitute for professional medical advice.")

with tab3:
    st.header("COVID-19 India Dashboard")
    covid_df = pd.read_csv("data/covid_india_snapshot.csv")
    metric_options = ['Total Cases', 'Active', 'Discharged', 'Deaths']
    selected_metric = st.selectbox("Select a metric to visualize:", options=metric_options)
    if selected_metric:
        df_sorted = covid_df.sort_values(by=selected_metric, ascending=False)
        fig = px.bar(df_sorted, x='State/UTs', y=selected_metric, title=f"State-wise Comparison of {selected_metric}")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(covid_df)