import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Machine Learning Model ---

# Cache the model training to prevent retraining on every interaction
@st.cache_resource
def train_symptom_checker_model():
    # Load the symptom dataset
    symptom_df = pd.read_csv("data/symptom_data.csv")
    
    # Define features (symptoms) and target (disease)
    features = ['fever', 'cough', 'headache', 'sore_throat', 'fatigue', 'body_aches']
    target = 'disease'
    
    X = symptom_df[features]
    y = symptom_df[target]
    
    # Split data for training and testing (optional for this simple case, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, features

# --- CORE CHATBOT FUNCTIONALITY (From Phase 2) ---

@st.cache_data
def load_qa_data():
    df = pd.read_csv("data/qa_data.csv")
    return df

def get_answer(user_question, df):
    user_question = user_question.lower()
    for index, row in df.iterrows():
        question_keywords = row['question'].lower().split()
        if any(word in user_question.split() for word in user_question.split()):
            return row['answer']
    return "I'm sorry, I don't have an answer for that. Please consult a medical professional for advice."

# --- STREAMLIT USER INTERFACE ---

st.title("AI Health Assistant")

# Train the ML model (or load from cache)
model, features = train_symptom_checker_model()

# Create tabs for different features
tab1, tab2 = st.tabs(["Chatbot", "Symptom Checker"])

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
        translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        answer = get_answer(translated_input, qa_df)
        translated_answer = GoogleTranslator(source='en', target=selected_language_code).translate(answer)
        st.write(translated_answer)

# --- Symptom Checker Tab ---
with tab2:
    st.header("Symptom Checker")
    st.write("Select the symptoms you are experiencing, and the AI will predict a possible condition.")
    
    # Create checkboxes for each symptom
    user_symptoms = {}
    for feature in features:
        user_symptoms[feature] = st.checkbox(feature.replace('_', ' ').title())
        
    # Predict button
    if st.button("Predict Condition"):
        # Prepare the user's input for the model
        input_data = [user_symptoms[feature] for feature in features]
        
        # Make a prediction
        prediction = model.predict([input_data])
        
        st.subheader("Prediction:")
        st.success(f"Based on your symptoms, the model predicts you might have: **{prediction[0]}**")
        st.warning("Disclaimer: This is an AI prediction and not a substitute for professional medical advice. Please consult a doctor.")