import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator

# --- CORE CHATBOT FUNCTIONALITY ---

# This function loads your Q&A data from the CSV file.
# @st.cache_data makes the app faster by loading the data only once.
@st.cache_data
def load_data():
    df = pd.read_csv("data/qa_data.csv")
    return df

# This function finds an answer in your data for the user's question based on keywords.
def get_answer(user_question, df):
    user_question = user_question.lower()
    for index, row in df.iterrows():
        question_keywords = row['question'].lower().split()
        if any(word in user_question.split() for word in question_keywords):
            return row['answer']
    return "I'm sorry, I don't have an answer for that. Please consult a medical professional for advice."

# --- STREAMLIT USER INTERFACE ---

st.title("AI Health Assistant")

# Load the Q&A data when the app starts.
qa_df = load_data()

# Create a language selection dropdown in the sidebar.
language_options = {'English': 'en', 'Hindi': 'hi', 'Marathi': 'mr'}
selected_language_name = st.sidebar.selectbox("Select Language", options=list(language_options.keys()))
selected_language_code = language_options[selected_language_name]

# Translate the UI text elements using the deep-translator library.
welcome_text = GoogleTranslator(source='auto', target=selected_language_code).translate("Hello! Ask me a health-related question.")
input_placeholder = GoogleTranslator(source='auto', target=selected_language_code).translate("Type your question here...")

st.write(welcome_text)

# Create a text input box for the user to ask a question.
user_input = st.text_input("", placeholder=input_placeholder)

# This block runs when the user types something and presses Enter.
if user_input:
    # Translate the user's question from their language to English so we can search our CSV.
    translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    
    # Find the answer in the dataframe.
    answer = get_answer(translated_input, qa_df)
    
    # Translate the English answer back to the user's selected language.
    translated_answer = GoogleTranslator(source='en', target=selected_language_code).translate(answer)
    
    # Display the final, translated answer.
    st.write(translated_answer)