
import streamlit as st
import pandas as pd
import numpy as np
import joblib # To save/load the trained model and preprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re
import emoji

# Define the clean_text function (copy from your notebook)
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = emoji.replace_emoji(text, replace=' ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_address_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = emoji.replace_emoji(text, replace=' ')
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text





# Load the trained model and preprocessor
# Make sure the paths are correct if you save them to Google Drive
try:
    loaded_model = joblib.load('D:/PythonFiles/MyProject/ProjectFinal/tiktok_comment_count_model.h5')
    # loaded_preprocessor = joblib.load('D:/PythonFiles/MyProject/preprocessor.pkl')
    st.success("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("Model or preprocessor file not found. Please make sure 'tiktok_comment_count_model.pkl' and 'preprocessor.pkl' exist.")
    st.stop() # Stop the app if files are not found

# Streamlit App Title
st.title("TikTok Comment Count Predictor")

# Input fields for the features
st.write("Fill all the fields")

# Text Features Input
st.subheader("Text Features:")
desc_input = st.text_area("Description", "")
challenges_input = st.text_area("Challenges (comma-separated)", "")
poi_name_input = st.text_input("POI Name", "")
music_title_input = st.text_input("Music Title", "")
music_author_name_input = st.text_input("Music Author Name", "")
address_input = st.text_area("Address", "")
# poi_tt_type_name_tiny_input = st.text_input("POI Type (Tiny)", "")


# Numerical Features Input
st.subheader("Numerical Features:")
collect_count_input = st.number_input("Collect Count", min_value=0)
digg_count_input = st.number_input("Digg Count", min_value=0)
play_count_input = st.number_input("Play Count", min_value=0)
share_count_input = st.number_input("Share Count", min_value=0)
vq_score_input = st.number_input("VQ Score", min_value=0.0)
duration_input = st.number_input("Duration (seconds)", min_value=0)
music_duration_input = st.number_input("Music Duration (seconds)", min_value=0.0)

# Categorical (Encoded) Features Input (based on your label encoding)
st.subheader("Categorical Features (Enter 0 or 1 based on your encoding):")
# duet_display_input = st.selectbox("Duet Display", [0, 1]) # Assuming 0 and 1 are your encoded values
duet_enabled_input = st.selectbox("Duet Enabled", ["-- Select --",0, 1]) # Assuming 0 and 1 are your encoded values
# is_ad_input = st.selectbox("Is Ad", [0, 1]) # Assuming 0 and 1 are your encoded values
# item_comment_status_input = st.number_input("Item Comment Status", min_value=0) # Assuming it's numerical or encoded
# item_mute_input = st.selectbox("Item Mute", [0, 1]) # Assuming 0 and 1 are your encoded values
item_control_can_repost_input = st.selectbox("Item Control Can Repost", ["-- Select --",0, 1]) # Assuming 0 and 1 are your encoded values
# official_item_input = st.selectbox("Official Item", [0, 1]) # Assuming 0 and 1 are your encoded values
# original_item_input = st.selectbox("Original Item", [0, 1]) # Assuming 0 and 1 are your encoded values
# share_enabled_input = st.selectbox("Share Enabled", [0, 1]) # Assuming 0 and 1 are your encoded values
# stitch_display_input = st.selectbox("Stitch Display", [0, 1]) # Assuming 0 and 1 are your encoded values
stitch_enabled_input = st.selectbox("Stitch Enabled", ["-- Select --",0, 1]) # Assuming 0 and 1 are your encoded values
# user_tt_seller_input = st.selectbox("User TT Seller", [0, 1]) # Assuming 0 and 1 are your encoded values
# user_verified_input = st.selectbox("User Verified", [0, 1]) # Assuming 0 and 1 are your encoded values
# diversification_id_input = st.number_input("Diversification ID", min_value=0.0) # Assuming it's numerical


# Create a button to predict
if st.button("Predict Comment Count"):
    # Create a dictionary with the input data
    input_data = {
        'desc': [desc_input],
        'challenges': [challenges_input],
        'poi_name': [poi_name_input],
        'music_title': [music_title_input],
        'music_author_name': [music_author_name_input],
        'address': [address_input],
        # 'poi_tt_type_name_tiny': [poi_tt_type_name_tiny_input],

        'collect_count': [collect_count_input],
        'digg_count': [digg_count_input],
        'play_count': [play_count_input],
        'share_count': [share_count_input],
        'vq_score': [vq_score_input],
        'duration': [duration_input],
        'music_duration': [music_duration_input],

        # 'duet_display': [duet_display_input],
        'duet_enabled': [duet_enabled_input],
        # 'is_ad': [is_ad_input],
        # 'item_comment_status': [item_comment_status_input],
        # 'item_mute': [item_mute_input],
        'item_control_can_repost': [item_control_can_repost_input],
        # 'official_item': [official_item_input],
        # 'original_item': [original_item_input],
        # 'share_enabled': [share_enabled_input],
        # 'stitch_display': [stitch_display_input],
        'stitch_enabled': [stitch_enabled_input],
        # 'user_tt_seller': [user_tt_seller_input],
        # 'user_verified': [user_verified_input],
        # 'diversification_id': [diversification_id_input],

         # Add other features here based on your model's training data
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame(input_data)

    # Apply the same text cleaning to the input text features
    input_df['clean_desc'] = input_df['desc'].apply(clean_text)
    input_df['clean_challenges'] = input_df['challenges'].apply(clean_text)
    input_df['clean_poi_name'] = input_df['poi_name'].apply(clean_text)
    input_df['clean_music_title'] = input_df['music_title'].apply(clean_text)
    input_df['clean_music_author_name'] = input_df['music_author_name'].apply(clean_text)
    input_df['clean_address'] = input_df['address'].apply(clean_address_text)
    # input_df['clean_poi_tt_type_name_tiny'] = input_df['poi_tt_type_name_tiny'].apply(clean_text)



    # Select features in the same order as trained
    # Make sure this list matches the order and names used in your ColumnTransformer
    feature_order = ['clean_desc', 'clean_challenges', 'clean_poi_name', 'clean_music_title','clean_music_author_name','clean_address'
                     # ,'clean_poi_tt_type_name_tiny'
                     ,'collect_count', 'digg_count', 'play_count', 'share_count','vq_score', 'duration', 'music_duration','music_original','stitch_enabled','item_control_can_repost','duet_enabled',
                      # 'duet_display',
                      # 'is_ad',
                      # 'item_comment_status', 'item_mute',
                      # 'official_item', 'original_item', 'share_enabled', 'stitch_display', 'user_tt_seller', 'user_verified',
                      # 'diversification_id'

                     ]

    # Ensure input_df has all required columns, even if empty, before applying preprocessor
    # This is crucial if your ColumnTransformer expects all original columns
    # Create a dummy DataFrame with all expected columns
    dummy_data = {col: [None] for col in feature_order} # Use None for object types, 0 or NaN for numbers
    dummy_df = pd.DataFrame(dummy_data)

    # Update dummy_df with actual input values
    for col in input_df.columns:
        if col in dummy_df.columns:
            dummy_df[col] = input_df[col]
        else:
             # Handle cases where the input column is new (like clean_text columns)
             # Ensure these are added correctly to the dummy_df
             if f'clean_{col}' in feature_order and f'clean_{col}' not in dummy_df.columns:
                 dummy_df[f'clean_{col}'] = input_df[col]

    if (
        not desc_input.strip() or
        not challenges_input.strip() or
        not poi_name_input.strip() or
        not music_title_input.strip() or
        not music_author_name_input.strip() or
        not address_input.strip() or
        collect_count_input == 0 or
        digg_count_input == 0 or
        play_count_input == 0 or
        share_count_input == 0 or
        vq_score_input == 0.0 or
        duration_input == 0 or
        music_duration_input == 0.0 or
        duet_enabled_input == "-- Select --" or
        item_control_can_repost_input == "-- Select --" or
        stitch_enabled_input == "-- Select --"
    ):
        st.warning("⚠️ Please fill all fields before predicting.")
    else:
        # Reorder dummy_df columns to match feature_order
        input_df_ordered = dummy_df[feature_order]
        # Preprocess the input data using the loaded preprocessor
        # processed_input = loaded_preprocessor.transform(input_df_ordered)

        input_df_ordered = input_df_ordered.fillna(0)
        # Make prediction
        prediction = loaded_model.predict(input_df_ordered)

        # Display the prediction
        st.subheader("Predicted Comment Count:")
        st.write(f"{prediction[0]:.2f}") # Display the prediction, formatted to 2 decimal places


# Run the Streamlit app
