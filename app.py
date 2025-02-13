import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import json
import os

# Load users from JSON file
def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}

# Save users to JSON file
def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

# Initialize users
if 'users' not in st.session_state:
    st.session_state['users'] = load_users()

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/osteoporosis_model.keras')
        class_names = np.load('model/class_names.npy', allow_pickle=True)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def welcome_page():
    st.title("Welcome to Osteoporosis Detection System")
    st.markdown("""
    ### Navigate through the application using the navigation bar.
    - **Prediction**: Analyze medical images.
    - **Prevention Tips**: Learn about osteoporosis prevention.
    - **About**: Information on osteoporosis.
    - **Graphs & Statistics**: View prediction statistics and confidence levels.
    """)

def signup_page():
    st.title("Create Account")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    name = st.text_input("Full Name")
    
    if st.button("Create Account"):
        if not new_username or not new_password or not name:
            st.error("All fields are required")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        elif new_username in st.session_state['users']:
            st.error("Username already exists")
        else:
            st.session_state['users'][new_username] = {
                'password': new_password,
                'name': name
            }
            save_users(st.session_state['users'])
            st.success("Account created successfully! Please sign in.")

def login_page():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Sign In"):
        users = st.session_state['users']
        if username in users and users[username]['password'] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def prediction_page():
    st.title("Image Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            model, class_names = load_model()
            if model is not None and class_names is not None:
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = float(np.max(prediction)) * 100
                st.markdown(f"""
                    **Result**: {predicted_class.title()}  
                    **Confidence**: {confidence:.2f}%
                """)
                
                if 'predictions' not in st.session_state:
                    st.session_state['predictions'] = []
                st.session_state['predictions'].append({
                    'result': predicted_class,
                    'confidence': confidence
                })

def statistics_page():
    st.title("Graphs & Statistics")
    if 'predictions' in st.session_state and st.session_state['predictions']:
        df = pd.DataFrame(st.session_state['predictions'])
        results_count = df['result'].value_counts()
        fig = px.pie(values=results_count.values, names=results_count.index, title='Distribution of Predictions')
        st.plotly_chart(fig)
    else:
        st.info("No predictions made yet.")

def prevention_tips():
    st.title("Prevention Tips")
    st.markdown("""
    ### Tips to Prevent Osteoporosis
    - **Eat a Calcium-Rich Diet**
    - **Exercise Regularly**
    - **Quit Smoking & Limit Alcohol**
    - **Get Regular Check-ups**
    """)

def main():
    st.set_page_config(page_title="Osteoporosis Detection", layout="wide")
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        page = st.sidebar.radio("Navigation", ["Home", "Sign In", "Sign Up"])
        if page == "Home":
            welcome_page()
        elif page == "Sign In":
            login_page()
        else:
            signup_page()
    else:
        page = st.sidebar.radio("Navigation", ["Welcome", "Prediction", "Prevention Tips", "Graphs & Statistics", "Logout"])
        if page == "Logout":
            st.session_state['logged_in'] = False
            st.rerun()
        elif page == "Prediction":
            prediction_page()
        elif page == "Graphs & Statistics":
            statistics_page()
        elif page == "Prevention Tips":
            prevention_tips()
        else:
            welcome_page()
        st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")

if __name__ == "__main__":
    main()
