import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotly.express as px # New import for plotting

# --- Machine Learning Model ---
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

# --- CORE CHATBOT FUNCTIONALITY ---
@st.cache_data
def load_qa_data():
    df = pd.read_csv("data/qa_data.csv")
    return df

def get_answer(user_question, df):
    user_question = user_question.lower()
    for index, row in df.iterrows():
        question_keywords = row['question'].lower().split()
        if any(word in user_question.split() for word in question_keywords):
            return row['answer']
    return "I'm sorry, I don't have an answer for that."

# --- STREAMLIT USER INTERFACE ---
st.title("AI Health Assistant")

model, features = train_symptom_checker_model()
# Add a new tab for the dashboard
tab1, tab2, tab3 = st.tabs(["Chatbot", "Symptom Checker", "Health Dashboard"])

# --- Chatbot Tab ---
with tab1:
    qa_df = load_qa_data()
    language_options = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
    selected_language_name = st.sidebar.selectbox("Select Language", options=list(language_options.keys()))
    selected_language_code = language_options[selected_language_name]

    welcome_text = GoogleTranslator(source='auto', target=selected_language_code).translate("Hello! Ask me a health-related question.")
    input_placeholder = GoogleTranslator(source='auto', target=selected_language_code).translate("Type your question here...")

    st.write(welcome_text)
    user_input = st.text_input("", placeholder=input_placeholder, key="chatbot_input")

    if user_input:
        critical_keywords = ['chest pain', 'breathing difficulty', 'suicide', 'emergency', 'heart attack', 'severe bleeding']
        translated_input_for_check = GoogleTranslator(source='auto', target='en').translate(user_input).lower()
        is_emergency = any(keyword in translated_input_for_check for keyword in critical_keywords)

        if is_emergency:
            emergency_title = GoogleTranslator(source='en', target=selected_language_code).translate("Emergency Situation Detected!")
            emergency_body = GoogleTranslator(source='en', target=selected_language_code).translate("Your message contains critical keywords. Please seek immediate medical help. Contact your local emergency services now.")
            st.error(f"**{emergency_title}**\n\n{emergency_body}")
        else:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
            answer = get_answer(translated_input, qa_df)
            translated_answer = GoogleTranslator(source='en', target=selected_language_code).translate(answer)
            st.write(translated_answer)

# --- Symptom Checker Tab ---
with tab2:
    st.header("Symptom Checker")
    st.write("Select the symptoms you are experiencing, and the AI will predict a possible condition.")
    user_symptoms = {}
    for feature in features:
        user_symptoms[feature] = st.checkbox(feature.replace('_', ' ').title())
        
    if st.button("Predict Condition"):
        input_data = [user_symptoms[feature] for feature in features]
        prediction = model.predict([input_data])
        st.subheader("Prediction:")
        st.success(f"Based on your symptoms, the model predicts you might have: **{prediction[0]}**")
        st.warning("Disclaimer: This is an AI prediction and not a substitute for professional medical advice. Please consult a doctor.")

# --- NEW: Health Dashboard Tab ---
with tab3:
    st.header("Dengue Cases Dashboard - Maharashtra")
    
    # Load the public health data
    dengue_df = pd.read_csv("data/public_dengue_stats.csv")
    
    # Allow users to filter by city
    cities = dengue_df['City'].unique()
    selected_cities = st.multiselect("Select cities to compare:", options=cities, default=cities)
    
    if selected_cities:
        # Filter the dataframe based on user selection
        filtered_df = dengue_df[dengue_df['City'].isin(selected_cities)]
        
        # Create an interactive bar chart
        fig = px.bar(filtered_df, x='Month', y='Cases', color='City', title="Monthly Dengue Cases by City")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one city to display the data.")