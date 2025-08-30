import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import warnings

# Suppress a common warning from sklearn for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --- Page Configuration ---
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺")

# --- API Configuration ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error("API Key not found or invalid. Please ensure you have a .streamlit/secrets.toml file with your GOOGLE_API_KEY.")
    st.stop()

# --- Gemini LLM Function ---
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

# --- Super Chatbot Tab ---
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

# --- Symptom Checker Tab ---
with tab2:
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

# --- COVID-19 Dashboard Tab (ENHANCED) ---
with tab3:
    st.header("COVID-19 India Dashboard")
    st.write("An interactive dashboard to visualize the COVID-19 status across states in India.")
    
    covid_df = pd.read_csv("data/covid_india_snapshot.csv")
    
    # --- NEW: Calculate new metrics ---
    covid_df['Discharge Ratio (%)'] = round((covid_df['Discharged'] / covid_df['Total Cases']) * 100, 2)
    covid_df['Death Ratio (%)'] = round((covid_df['Deaths'] / covid_df['Total Cases']) * 100, 2)

    # --- NEW: Add the new metrics to the dropdown options ---
    metric_options = ['Total Cases', 'Active', 'Discharged', 'Deaths', 'Discharge Ratio (%)', 'Death Ratio (%)']
    selected_metric = st.selectbox("Select a metric to visualize:", options=metric_options)
    
    if selected_metric:
        df_sorted = covid_df.sort_values(by=selected_metric, ascending=False)
        
        fig = px.bar(df_sorted, 
                     x='State/UTs', 
                     y=selected_metric, 
                     title=f"State-wise Comparison of {selected_metric}",
                     labels={'State/UTs': 'State / Union Territory'})
        
        fig.update_layout(xaxis_title="State / Union Territory", yaxis_title=f"Value for {selected_metric}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(covid_df)