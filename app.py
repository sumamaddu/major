import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import json
import os
from fpdf import FPDF

# Set page config first
st.set_page_config(page_title="Osteoporosis Detection", layout="wide")

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

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""
if 'users' not in st.session_state:
    st.session_state['users'] = load_users()

# Preprocess image for the model
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

def get_prevention_tips(confidence):
    if confidence > 80:
        return """
        - **Eat a Calcium-Rich Diet**
        - **Exercise Regularly**
        - **Quit Smoking & Limit Alcohol**
        - **Get Regular Check-ups**
        """
    elif confidence > 50:
        return """
        - **Consider Regular Bone Density Tests**
        - **Ensure Adequate Vitamin D Intake**
        """
    else:
        return """
        - **Consult with a healthcare provider**
        - **Consider lifestyle changes to reduce bone loss**
        """

def welcome_page():
    if st.session_state['logged_in']:
        st.title(f"Welcome back, {st.session_state['user_name']}!")
        st.markdown("""
        ### Now you can analyze X-ray images and view detailed reports.
        - **Prediction**: Analyze spine X-ray images.
        - **Prevention Tips**: Get tips based on your predictions.
        - **Graphs & Statistics**: See your prediction history and confidence levels.
        """)
    else:
        st.title("Welcome to Osteoporosis Detection System")
        st.markdown("""
        ### What is Osteoporosis?
        Osteoporosis is a condition where bones become weak and brittle, increasing the risk of fractures. This system uses advanced image analysis techniques to detect early signs of osteoporosis through spine X-rays.

        ### Navigate through the application using the navigation bar:
        - **Prediction**: Analyze spine X-ray images to detect osteoporosis.
        - **Prevention Tips**: Learn how to prevent osteoporosis with lifestyle changes.
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
            st.session_state['user_name'] = users[username]['name']
            st.session_state['username'] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    # Password reset link (direct reset option)
    if st.button("Forgot Password"):
        st.session_state['forgot_password'] = True
        st.rerun()

def reset_password_page():
    st.title("Reset Password")
    
    username_to_reset = st.text_input("Enter your username to reset password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")
    
    if st.button("Reset Password"):
        if username_to_reset not in st.session_state['users']:
            st.error("Username does not exist")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        else:
            # Update the password for the given username
            st.session_state['users'][username_to_reset]['password'] = new_password
            save_users(st.session_state['users'])
            st.success("Your password has been reset successfully!")
            st.session_state['forgot_password'] = False
            st.rerun()

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
                tips = get_prevention_tips(confidence)
                st.markdown(f"""
                    **Result**: {predicted_class.title()}  
                    **Confidence**: {confidence:.2f}%
                    {tips}
                """)
                
                if 'predictions' not in st.session_state:
                    st.session_state['predictions'] = []
                st.session_state['predictions'].append({
                    'result': predicted_class,
                    'confidence': confidence
                })
                
                # PDF Report Generation
                generate_pdf_report(predicted_class, confidence, tips)

def generate_pdf_report(predicted_class, confidence, tips):
    # Create PDF document
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Osteoporosis Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Result: {predicted_class.title()}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Prevention Tips:\n{tips}")
    
    # Save the PDF to a file
    pdf_file = "osteoporosis_report.pdf"
    pdf.output(pdf_file)

    # Provide a download link for the PDF
    with open(pdf_file, "rb") as f:
        st.download_button("Download Report", f, file_name=pdf_file, mime="application/pdf")

def statistics_page():
    st.title("Graphs & Statistics")
    if 'predictions' in st.session_state and st.session_state['predictions']:
        df = pd.DataFrame(st.session_state['predictions'])
        
        # Prediction Distribution (Pie Chart with Custom Colors)
        results_count = df['result'].value_counts()
        pie_colors = ['#ff9999', '#66b3ff', '#99ff99']  # Custom colors for pie chart
        pie_fig = px.pie(results_count, title="Prediction Distribution", names=results_count.index, values=results_count.values, color_discrete_sequence=pie_colors)
        st.plotly_chart(pie_fig)
        
        # Bar Chart for Results (With Custom Colors)
        bar_colors = ['#ffcc99', '#ff6666', '#ff3399']  # Custom colors for bar chart
        bar_fig = px.bar(results_count, title="Bar Chart of Prediction Results", color=results_count.index, color_discrete_sequence=bar_colors)
        st.plotly_chart(bar_fig)

        # Scatter Plot for Result vs. Confidence (With Custom Colors)
        scatter_fig = px.scatter(df, x="result", y="confidence", title="Prediction vs. Confidence", color="result", color_discrete_map={'Normal': '#66b3ff', 'Osteoporosis': '#ff6666'})
        st.plotly_chart(scatter_fig)
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
    if 'forgot_password' in st.session_state and st.session_state['forgot_password']:
        page = st.sidebar.radio("Navigation", ["Reset Password", "Back to Login"])
        if page == "Reset Password":
            reset_password_page()
        else:
            st.session_state['forgot_password'] = False
            st.rerun()
    elif not st.session_state['logged_in']:
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
            st.session_state['user_name'] = ""
            st.rerun()
        elif page == "Welcome":
            welcome_page()
        elif page == "Prediction":
            prediction_page()
        elif page == "Prevention Tips":
            prevention_tips()
        elif page == "Graphs & Statistics":
            statistics_page()

if __name__ == '__main__':
    main()
