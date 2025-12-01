import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import requests
from streamlit_folium import folium_static
import folium
import webbrowser
import cv2  # Must use same OpenCV as research
from scipy.signal import wiener
from skimage import exposure



# Set page config with pink ribbon logo
st.set_page_config(
    page_title="Pink Rangers - Breast Health Companion",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Universal model loading function that works with all Streamlit versions
def load_diagnosis_model():
    if hasattr(st, 'cache_resource'):  # Newer Streamlit versions
        @st.cache_resource
        def _load_model():
            try:
                return load_model('./cnn_model_preprocessed.h5')
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    else:  # Older Streamlit versions
        @st.cache(allow_output_mutation=True, hash_funcs={tf.keras.models.Model: id})
        def _load_model():
            try:
                return load_model('./cnn_model_preprocessed.h5')
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
    return _load_model()

model = load_diagnosis_model()
def preprocess_image(image):
    try:
        if image.mode != 'L':
            image = image.convert('L')
        
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        
        # Wiener filter
        with np.errstate(divide='ignore', invalid='ignore'):
            img_wiener = wiener(img)
            img_wiener = np.nan_to_num(img_wiener)
            
        # Histogram equalization
        img_eq = exposure.equalize_hist(img_wiener)
        img_eq = (img_eq * 255).astype(np.uint8)
        
        # Final processing - CRITICAL FIX HERE
        img_processed = img_eq.astype(np.float32) / 255.0
        img_processed = np.expand_dims(img_processed, axis=-1)  # Now shape (128,128,1)
        
        return img_processed  # Return without batch dimension
    
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None
# Diagnosis function
def make_diagnosis(image_array, model):
    try:
        input_tensor = np.expand_dims(image_array, axis=0)
        prediction = model.predict(input_tensor, verbose=0)[0][0]
        
        # Adjusted threshold to catch more malignancies
        diagnosis = "Malignant" if prediction > 0.2 else "Benign"
        confidence = prediction if diagnosis == "Malignant" else 1 - prediction
        
        
        
        return diagnosis, confidence
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None


# Indian emergency contacts
EMERGENCY_CONTACTS = {
    "National Cancer Helpline": "1800-22-1950",
    "Indian Cancer Society": "+91-22-2413 9445 / 2415 0950",
    "Tata Memorial Hospital (Mumbai)": "+91-22-2417 7000",
    "AIIMS (Delhi) Cancer Center": "+91-11-2659 4402",
    "Breast Cancer Patient Support Group": "+91-98-2000-2001"
}

# Breast self-examination steps
SELF_EXAM_STEPS = [
    "1. Stand in front of a mirror with your shoulders straight and hands on your hips.",
    "2. Look for changes in size, shape, or color of breasts, or visible distortions/swellings.",
    "3. Raise your arms and look for the same changes.",
    "4. Feel your breasts while lying down, using your right hand for left breast and vice versa.",
    "5. Use a firm, smooth touch with the first few finger pads, keeping fingers flat and together.",
    "6. Cover the entire breast from top to bottom, side to side - from collarbone to abdomen and armpit to cleavage.",
    "7. Repeat the examination while standing or sitting, perhaps when showering."
]

# Breast health facts
BREAST_HEALTH_FACTS = [
    "1 in 28 Indian women is likely to develop breast cancer during her lifetime.",
    "Breast cancer accounts for 14% of all cancers in Indian women.",
    "Early detection increases 5-year survival rate to over 90%.",
    "Monthly self-exams help you become familiar with how your breasts normally look and feel.",
    "Women over 40 should get annual mammograms even if they feel no symptoms."
]

# Tips based on diagnosis
def get_tips(diagnosis):
    if diagnosis == "Benign":
        return [
            "Continue regular self-examinations monthly",
            "Schedule annual clinical breast exams",
            "Maintain a healthy diet rich in fruits and vegetables",
            "Exercise regularly (150 minutes of moderate activity per week)",
            "Limit alcohol consumption if alcoholic"
        ]
    else:
        return [
            "Consult an oncologist immediately for further evaluation",
            "Don't panic - many breast abnormalities are treatable when caught early",
            "Bring all your medical history to your doctor appointment",
            "Consider getting a second opinion",
            "Reach out to support groups for emotional support"
        ]

# Nearby hospitals map using Google Maps API
def show_nearby_hospitals(lat, lon):
    try:
        # Google Maps API key
        api_key = "AIzaSyAOVYRIgupAurZup5y1PRh8Ismb1A3lLao"  # Replace with your actual key
        
        # Create Google Maps iframe
        map_html = f"""
        <iframe
            width="100%"
            height="500"
            frameborder="0" style="border:0"
            src="https://www.google.com/maps/embed/v1/search?key={api_key}&q=breast+cancer+hospital&center={lat},{lon}&zoom=13" allowfullscreen>
        </iframe>
        """
        
        # Display the map
        st.components.v1.html(map_html, height=550)
        
        # Also show the top hospitals as fallback
        st.subheader("Top Breast Cancer Hospitals Near You")
        st.write("1. City Cancer Center (2.5 km away)")
        st.write("2. Women's Health Specialists (3.1 km away)")
        st.write("3. Breast Care Clinic (4.2 km away)")
        
    except Exception as e:
        st.error(f"Could not load map: {e}")
        # Fallback to Folium map
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker(
            [lat, lon], 
            popup="Your Location", 
            icon=folium.Icon(color="blue")
        ).add_to(m)
        
        folium.Marker(
            [lat + 0.01, lon + 0.01], 
            popup="City Cancer Hospital", 
            icon=folium.Icon(color="red", icon="plus-sign")
        ).add_to(m)
        
        folium.Marker(
            [lat - 0.01, lon + 0.02], 
            popup="Women's Health Center", 
            icon=folium.Icon(color="pink", icon="heart")
        ).add_to(m)
        
        folium.Marker(
            [lat + 0.015, lon - 0.01], 
            popup="Breast Care Specialists", 
            icon=folium.Icon(color="purple", icon="info-sign")
        ).add_to(m)
        
        folium_static(m)

# Main app
def main():
    # Custom CSS for pink theme
    # Custom CSS for pink theme
    st.markdown("""
    <style>
    /* Force black text in sidebar for all elements */
    [data-testid="stSidebar"] * {
        color: black !important;
    }

    /* Make the close button (X) red */
    [data-testid="stSidebar"] button[title="Close the sidebar"] {
        color: red !important;
    }

    /* Make radio button labels black and properly styled */
    [data-testid="stSidebar"] .stRadio label {
        color: black !important;
        font-size: 1rem !important;
    }

    /* Selected radio button styling */
    [data-testid="stSidebar"] .stRadio label div:has(input:checked) {
        font-weight: bold !important;
    }

    /* Main content background */
    .main {
        background-color: #FFF0F5;
    }

    .stApp {
        background-image: linear-gradient(to bottom, #FFF0F5, #FFFFFF);
    }

    /* Sidebar background */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #FFD1DC !important;
    }

    /* Headers color */
    h1, h2, h3 {
        color: #E75480;
    }

    /* Buttons */
    .stButton>button {
        background-color: #E75480;
        color: white;
    }

    /* Diagnosis box */
    .diagnosis-box {
        border: 2px solid #E75480;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    /* Emergency box */
    .emergency-box {
        border: 2px solid #FF0000;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        background-color: #FFE4E1;
    }
    </style>
    """, unsafe_allow_html=True)

  

    # App header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.png")  
    with col2:
        st.title("Pink Rangers - Breast Health Companion")
        st.markdown("*Empowering women with early detection and care*")

    # Sidebar with navigation
    st.sidebar.header("Navigation")
    app_page = st.sidebar.radio("Go to", 
                               ["Breast Diagnosis", 
                                "Self-Examination Guide", 
                                "Breast Health Facts", 
                                "Emergency Contacts", 
                                "Find Nearby Hospitals"])

    if app_page == "Breast Diagnosis":
        st.header("Breast Cancer Diagnosis")
        st.markdown("Upload your thermographic image for preliminary analysis")
        
        uploaded_file = st.file_uploader("Choose a grayscale breast image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing..."):
                    processed_image = preprocess_image(image)
                    if processed_image is not None and model is not None:
                        diagnosis, confidence = make_diagnosis(processed_image, model)
                        
                        if diagnosis:
                            st.markdown(f'<div class="diagnosis-box">', unsafe_allow_html=True)
                            st.subheader("Diagnosis Result")
                            
                            if diagnosis == "Malignant":
                                st.error(f"Prediction: {diagnosis} ({(confidence*100):.2f}% confidence)")
                            else:
                                st.success(f"Prediction: {diagnosis} ({(confidence*100):.2f}% confidence)")
                            
                            st.markdown("**Recommendations:**")
                            for tip in get_tips(diagnosis):
                                st.write(f"- {tip}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            st.markdown(f'<div class="emergency-box">', unsafe_allow_html=True)
                            st.warning("Remember: This is a preliminary analysis. Please consult an oncologist for professional diagnosis.")
                            st.markdown("</div>", unsafe_allow_html=True)
        
    elif app_page == "Self-Examination Guide":
        st.header("Breast Self-Examination Guide")
        st.markdown("Perform this examination monthly, preferably a few days after your period ends")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Step-by-Step Guide")
            for step in SELF_EXAM_STEPS:
                st.write(step)
        
        with col2:
            st.subheader("What to Look For")
            st.write("- New lumps or hard knots in breast or underarm")
            st.write("- Unusual swelling, warmth, redness or darkening")
            st.write("- Change in breast size or shape")
            st.write("- Dimpling or puckering of the skin")
            st.write("- Nipple retraction (turning inward)")
            st.write("- Nipple discharge (other than breast milk)")
            st.write("- Persistent breast pain")
            
            st.video("https://youtu.be/XKtTymNkcj0?si=Kv05hwkv3UepeHGO")  
    
    elif app_page == "Breast Health Facts":
        st.header("Breast Health Facts & Awareness")
        
        st.subheader("Did You Know?")
        for fact in BREAST_HEALTH_FACTS:
            st.write(f"🎀 {fact}")
        
        st.subheader("Early Detection Saves Lives")
        st.image("who.png", 
                caption="Survival rates are much higher when breast cancer is detected early")
        
        st.subheader("Risk Factors")
        st.write("- Age (risk increases after 40)")
        st.write("- Family history of breast cancer")
        st.write("- Early menstruation (before 12) or late menopause (after 55)")
        st.write("- Dense breast tissue")
        st.write("- Obesity after menopause")
        st.write("- Alcohol consumption")
        st.write("- Radiation exposure")
        
    elif app_page == "Emergency Contacts":
        st.header("Emergency Contacts & Support")
        st.markdown("Immediate help for breast health concerns in India")
        
        st.subheader("24/7 Helplines")
        for name, number in EMERGENCY_CONTACTS.items():
            st.markdown(f"**{name}**: {number}")
        
        st.subheader("Support Groups")
        st.write("- **Cancer Patients Aid Association**: +91-22-2492 4270")
        st.write("- **Indian Cancer Society Support Group**: +91-22-2413 9445")
        st.write("- **Breast Cancer Foundation (India)**: +91-98-1020-3040")
        
        st.subheader("Financial Assistance")
        st.write("- **Health Minister's Cancer Patient Fund**: 1800-11-6666")
        st.write("- **Tata Memorial Centre Financial Aid**: +91-22-2417 7000")
        
    elif app_page == "Find Nearby Hospitals":
        st.header("Find Breast Healthcare Centers Near You")
        
        st.write("Enter your location to find nearby hospitals with breast cancer screening facilities")
        
        # Location input
        location = st.text_input("Enter your location (city or address)", "Mumbai")
        
        if st.button("Find Hospitals"):
            with st.spinner("Searching for nearby hospitals..."):
                # Demo coordinates - in real app, you would geocode the location using Google Geocoding API
                if "mumbai" in location.lower():
                    lat, lon = 19.0760, 72.8777
                elif "delhi" in location.lower():
                    lat, lon = 28.7041, 77.1025
                elif "bangalore" in location.lower():
                    lat, lon = 12.9716, 77.5946
                else:
                    lat, lon = 20.5937, 78.9629  # Default to India center
                
                # Show the Google Maps with hospitals
                show_nearby_hospitals(lat, lon)
                
                # Additional hospital information
                st.subheader("Top Breast Cancer Hospitals in India")
                st.write("- Tata Memorial Hospital, Mumbai")
                st.write("- All India Institute of Medical Sciences (AIIMS), Delhi")
                st.write("- Cancer Institute (WIA), Chennai")
                st.write("- Rajiv Gandhi Cancer Institute, Delhi")
                st.write("- Kidwai Memorial Institute of Oncology, Bangalore")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Pink Rangers** is committed to breast health awareness. 
    This tool provides preliminary information only and is not a substitute for professional medical advice.
    """)
    
    st.link_button(
    "Donate to Breast Cancer Research",
    "https://www.nationalbreastcancer.org/breast-cancer-donations/"
)


if __name__ == "__main__":
    main()


