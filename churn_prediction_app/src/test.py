import streamlit as st
import google.generativeai as genai
import os

# Configure the Google Generative AI client with the API key from Streamlit secrets
try:
    # Access the API key using st.secrets
    # If running locally, make sure .streamlit/secrets.toml exists in your project root
    # If deploying to Streamlit Community Cloud, secrets are managed via the UI
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except AttributeError:
    st.error("GEMINI_API_KEY not found in Streamlit secrets. Please configure it.")
    st.stop() # Stop the app execution if key is missing

# Now, try to list models to confirm API access and find available models
available_models = []
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            available_models.append(m.name)

    if not available_models:
        st.error("No models supporting 'generateContent' found with your API key.")
        st.stop()

    # You can print them to the Streamlit app for debugging
    # st.write("Available Gemini Models:", available_models)

    # Prioritize widely used models, or pick the first available if multiple
    if 'gemini-1.5-pro-latest' in available_models:
        GEMINI_MODEL_NAME = 'gemini-1.5-pro-latest'
    elif 'gemini-1.0-pro' in available_models:
        GEMINI_MODEL_NAME = 'gemini-1.0-pro'
    elif available_models:
        # Fallback to the first available model if preferred ones aren't there
        GEMINI_MODEL_NAME = available_models[0]
    else:
        st.error("No suitable Gemini model found.")
        st.stop()

    # Initialize your model using the chosen name
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    st.success(f"Successfully connected to Gemini model: {GEMINI_MODEL_NAME}")

except Exception as e:
    st.error(f"Could not connect to Gemini or list models. Error: {e}")
    st.info("Please check your API key and network connection.")
    st.stop()

# The rest of your Streamlit app code goes here,
# now assuming 'model' is successfully initialized.
# For example, if you use it in predict_churn
# from model_predictor import load_all_models, predict_churn
# Your `predict_churn` function would then need to accept 'model'
# or the model would be initialized within that module after configuring genai.