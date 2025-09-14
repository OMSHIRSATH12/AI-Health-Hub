# 🩺 AI Health Hub

### An advanced, feature-rich, AI-powered health platform built for your hackathon.

---

## 🚀 Live Demo

**[Click here to interact with the live, deployed application!](https://ai-health-app-n5n3krmbtznpvay5safhfj.streamlit.app/)**

## 📖 Overview

AI Health Hub is an advanced, all-in-one health platform designed to provide users with a suite of intelligent, real-time tools. Moving beyond a simple chatbot, this application functions as a comprehensive health companion, integrating a generative AI, a machine learning predictor, live data dashboards, and a wealth of health information into a single, easy-to-use, and beautifully designed interface.

## ✨ Key Features

This project is packed with a wide range of "dominating" features that make it a true health hub:

* **💬 Conversational AI Super Chatbot:**
    * Powered by the **Google Gemini API** for intelligent, human-like conversations.
    * Features **conversational memory** to understand follow-up questions.
    * Includes a strict **on-topic guardrail** to ensure it only answers health-related queries.
    * **Multilingual support** for users to interact in their native language.

* **🗣️ Voice Assistant:**
    * **Speech-to-Text:** An "Ask with Voice" button for hands-free queries.
    * **Text-to-Speech:** The AI's responses are spoken aloud for enhanced accessibility.

* **🚨 Emergency Alert System:**
    * Intelligently detects critical keywords to differentiate between informational queries and real emergencies.
    * Provides a high-impact **visual red alert** and an **audible beep** for emergencies.
    * Integrates with a **free, open-source mapping service** to display nearby hospitals when an emergency is detected.

* **🔍 Personalized Disease Predictor:**
    * Uses a **Scikit-learn Machine Learning model** trained on a synthetically augmented dataset.
    * Provides predictions based on **age, gender, and a wide range of symptoms**.
    * Includes **AI-generated recommendations** for common medicines and diet after a prediction.

* **📄 AI Medical Report Summarizer:**
    * Allows users to upload a `.txt` file of a complex medical report.
    * Uses the Gemini API to generate a simple, easy-to-understand summary.

* **📊 Interactive Data Dashboards:**
    * Visualizes a real-world **COVID-19 dataset** for India with interactive Plotly charts.
    * Includes calculated metrics like "Discharge Ratio" and "Death Ratio".

* **📰 Live Health News & Alerts:**
    * A **Health News Feed** that uses a smart, multi-keyword query to pull highly relevant articles from the NewsAPI.
    * A **Health Hub** that displays live **Disease Outbreak Alerts** directly from the World Health Organization (WHO).

## 🛠️ Tech Stack

* **Core:** Python, Streamlit
* **Generative AI:** Google Gemini API
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Data Visualization:** Plotly, Folium (for maps)
* **External APIs:** NewsAPI, OpenStreetMap, WHO Disease Outbreak News API
* **Voice & Audio:** SpeechRecognition, gTTS
* **Geolocation:** Geopy

## ⚡ How to Run Locally

1.  **Clone the repository.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create your secrets file:**
    * Create a folder named `.streamlit`.
    * Inside it, create a file named `secrets.toml`.
    * Add your API keys to this file in the following format:
        ```toml
        GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
        NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE"
        ```
5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
