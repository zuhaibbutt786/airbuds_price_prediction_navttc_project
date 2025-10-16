import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model pipeline
try:
    model_pipeline = joblib.load('best_airbuds_price_predictor.joblib')
except FileNotFoundError:
    st.error("Error: The model file 'best_airbuds_price_predictor.joblib' was not found.")
    st.info("Please make sure you have run the training script first to create the model file.")
    st.stop()

# --- Helper Function for Cleaning Inputs (Matches Training Preprocessing) ---
# This is crucial to ensure consistent data format between training and prediction
def clean_input_value(feature_name, value):
    if pd.isna(value) or value == '' or value == 'N/A' or value == 'Unknown':
        return np.nan # Use NaN for numerical imputer
    
    # Specific cleaning logic matching the training script
    if feature_name in ['General Features - Driver Size', 'Connectivity - Bluetooth Version',
                        'Connectivity - Bluetooth Range', 'Battery - Capacity for buds',
                        'Battery - Capacity for Case', 'Battery - Playtime', 'Battery - Charging Time']:
        # This part should mimic the extract_numeric function from the training script
        text = str(value).lower()
        if 'v' in text: 
            text = text.replace('v', '')
        if 'hrs' in text:
            parts = text.split('-')
            if len(parts) > 1:
                try: return (float(parts[0].strip()) + float(parts[1].replace('hrs', '').strip())) / 2
                except ValueError: return np.nan
            else:
                try: return float(text.replace('hrs', '').strip())
                except ValueError: return np.nan
        if 'hours' in text:
            parts = text.split('-')
            if len(parts) > 1:
                try: return (float(parts[0].strip()) + float(parts[1].replace('hours', '').strip())) / 2
                except ValueError: return np.nan
            else:
                try: return float(text.replace('hours', '').strip())
                except ValueError: return np.nan
        if 'm' in text and 'mm' not in text:
            try: return float(text.replace('m', '').strip())
            except ValueError: return np.nan
        if 'ft' in text:
            try: return float(text.replace('ft', '').strip()) * 0.3048
            except ValueError: return np.nan
        if 'mah' in text:
            try: return float(text.replace('mah', '').strip())
            except ValueError: return np.nan
        if 'mm' in text:
            try: return float(text.replace('mm', '').strip())
            except ValueError: return np.nan
        try: return float(text)
        except ValueError: return np.nan
    
    elif feature_name in ['General Features - Noise Cancellation', 'General Features - Water Resistant',
                          'General Features - Auto Pairing', 'General Features - Mic', 'Connectivity - Microphone']:
        s_value = str(value).lower().strip()
        if s_value in ['yes', 'y', 'true', 'anc', 'ai call noise cancelation', 'enc', 'dual-mic noise reduction', 'dust, sweat, and water resistant5']:
            return 'Yes'
        elif s_value in ['no', 'n', 'false']:
            return 'No'
        else:
            return 'Unknown'
            
    elif feature_name == 'General Features - Charging Interface':
        s_value = str(value).lower().strip()
        if 'type c' in s_value or 'usb-c' in s_value or 'c-type' in s_value:
            return 'Type-C'
        elif 'micro usb' in s_value or 'micro' in s_value:
            return 'Micro USB'
        elif 'lightning' in s_value:
            return 'Lightning'
        else:
            return 'Unknown'
            
    elif feature_name == 'General Features - Compatibility':
        s_value = str(value).lower()
        if 'android' in s_value and 'ios' in s_value:
            return 'Android & iOS'
        elif 'android' in s_value:
            return 'Android Only'
        elif 'ios' in s_value:
            return 'iOS Only'
        elif 'windows' in s_value:
            return 'Windows Compatible'
        else:
            return 'Unknown'
            
    return value # For other text fields, return as is (if any)

# --- Streamlit App ---

st.set_page_config(
    page_title="Airbuds Price Predictor",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a fancier look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background-color: #e0e5ec;
    }
    h1, h2, h3 {
        color: #333333;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.3);
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        border: 2px solid #4CAF50;
    }
    .prediction-text {
        font-size: 2.5em;
        color: #007bff;
        font-weight: bold;
        margin-top: 10px;
    }
    .placeholder-text {
        color: #888888;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 Airbuds Price Predictor 💰")
st.markdown("<h3>Predict the price of airbuds based on their features!</h3>", unsafe_allow_html=True)

st.write("---")

# --- Sidebar for User Inputs ---
st.sidebar.header("Earbud Specifications")

# Input fields
user_inputs = {}

# Define options for dropdowns and help messages
feature_details = {
    'General Features - Noise Cancellation': {
        'type': 'dropdown', 'options': ['Unknown', 'Yes', 'No'], 'default': 'Unknown',
        'help': "Does the earbud feature Noise Cancellation (ANC/ENC)? Select 'Yes' or 'No'."
    },
    'General Features - Water Resistant': {
        'type': 'dropdown', 'options': ['Unknown', 'Yes', 'No'], 'default': 'Unknown',
        'help': "Is the earbud water resistant (e.g., IPX4, IPX5, IP67)? Select 'Yes' or 'No'."
    },
    'General Features - Charging Interface': {
        'type': 'dropdown', 'options': ['Unknown', 'Type-C', 'Micro USB', 'Lightning'], 'default': 'Type-C',
        'help': "What type of charging interface does the earbud case use? (e.g., Type-C, Micro USB, Lightning)"
    },
    'General Features - Auto Pairing': {
        'type': 'dropdown', 'options': ['Unknown', 'Yes', 'No'], 'default': 'Yes',
        'help': "Does the earbud support automatic pairing with devices? Select 'Yes' or 'No'."
    },
    'General Features - Compatibility': {
        'type': 'dropdown', 'options': ['Unknown', 'Android & iOS', 'Android Only', 'iOS Only', 'Windows Compatible'], 'default': 'Android & iOS',
        'help': "What operating systems are the earbuds compatible with?"
    },
    'General Features - Mic': {
        'type': 'dropdown', 'options': ['Unknown', 'Yes', 'No'], 'default': 'Yes',
        'help': "Do the earbuds have a built-in microphone? Select 'Yes' or 'No'."
    },
    'General Features - Driver Size': {
        'type': 'number_input', 'min_value': 0.0, 'max_value': 20.0, 'default': 10.0,
        'help': "The size of the earbud's audio driver in millimeters (e.g., 10mm, 13mm)."
    },
    'Connectivity - Bluetooth Version': {
        'type': 'number_input', 'min_value': 3.0, 'max_value': 6.0, 'default': 5.0, 'step': 0.1,
        'help': "The Bluetooth version of the earbuds (e.g., 5.0, 5.2, 5.3)."
    },
    'Connectivity - Bluetooth Range': {
        'type': 'number_input', 'min_value': 1.0, 'max_value': 30.0, 'default': 10.0,
        'help': "The effective Bluetooth range in meters (e.g., 10m)."
    },
    'Connectivity - Microphone': {
        'type': 'dropdown', 'options': ['Unknown', 'Yes', 'No'], 'default': 'Yes',
        'help': "Does the earbud specifically list a microphone for connectivity features? (Often redundant with 'General Features - Mic', but good to include if present in data.)"
    },
    'Battery - Capacity for buds': {
        'type': 'number_input', 'min_value': 0.0, 'max_value': 100.0, 'default': 40.0,
        'help': "Battery capacity of each earbud in mAh (e.g., 30mAh, 50mAh)."
    },
    'Battery - Capacity for Case': {
        'type': 'number_input', 'min_value': 0.0, 'max_value': 5000.0, 'default': 300.0,
        'help': "Battery capacity of the charging case in mAh (e.g., 300mAh, 2000mAh)."
    },
    'Battery - Playtime': {
        'type': 'number_input', 'min_value': 0.0, 'max_value': 60.0, 'default': 4.0,
        'help': "Total playtime on a single charge for earbuds, in hours (e.g., 3-4 Hrs, 5-6 Hours)."
    },
    'Battery - Charging Time': {
        'type': 'number_input', 'min_value': 0.0, 'max_value': 5.0, 'default': 1.5,
        'help': "Time required to fully charge the earbuds/case in hours (e.g., 1.5 Hrs, 2 Hours)."
    }
}

for feature, details in feature_details.items():
    label = feature.replace('General Features - ', '').replace('Connectivity - ', '').replace('Battery - ', '')
    
    if details['type'] == 'dropdown':
        user_inputs[feature] = st.sidebar.selectbox(
            label,
            options=details['options'],
            index=details['options'].index(details['default']) if details['default'] in details['options'] else 0,
            help=details['help']
        )
    elif details['type'] == 'text_input':
         user_inputs[feature] = st.sidebar.text_input(
            label,
            placeholder=details['placeholder'],
            help=details['help']
        )
    elif details['type'] == 'number_input':
        user_inputs[feature] = st.sidebar.number_input(
            label,
            min_value=details['min_value'],
            max_value=details['max_value'],
            value=details['default'],
            step=details.get('step', 1.0),
            help=details['help']
        )

# --- Help Section (Main Page) ---
with st.expander("❓ **Need Help with Input Fields?**"):
    st.markdown("Here's a guide to what each input field means:")
    for feature, details in feature_details.items():
        st.markdown(f"**{feature.replace('General Features - ', '').replace('Connectivity - ', '').replace('Battery - ', '')}:** {details['help']}")

# --- Prediction Button ---
if st.button("Predict Price", key="predict_button"):
    # Create a DataFrame from user inputs
    # Apply cleaning function to ensure consistency with training data
    processed_inputs = {col: clean_input_value(col, val) for col, val in user_inputs.items()}
    input_df = pd.DataFrame([processed_inputs])

    # Make prediction
    try:
        prediction = model_pipeline.predict(input_df)[0]
        
        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>Predicted Price:</h2>
                <div class="prediction-text">Rs {prediction:,.2f}</div>
                <p style="color: #666;">*This is an estimated price based on the provided features.</p>
            </div>
            """, unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values. Some combinations might lead to issues if they are significantly outside the training data range.")

st.write("---")
st.markdown("Created with ❤️ for Airbuds Enthusiasts")
