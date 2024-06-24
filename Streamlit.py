import streamlit as st
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)

def load_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"Model file not found: {file_path}")
        return None
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        st.error(f"File not found: {fnf_error}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
    return None

def main():
    st.title("Model Inference with Streamlit")
    
    model = load_model('best_clf (1).pkl')
    if model is None:
        return
    
    # Example of using the loaded model
    st.write("Model loaded successfully!")
    # Add further code to use the model for predictions, etc.

if __name__ == "__main__":
    main()
