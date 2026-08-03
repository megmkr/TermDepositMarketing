
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API
import xgboost as xgb # Import xgboost to load native XGBoost models
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Initialize Flask app with a name
super_subscribe_api = Flask("super_subscribe_api") #code to define the name of the app

# Function to clean feature names from ColumnTransformer output
def clean_feature_names(feature_names):
    cleaned_names = []
    for name in feature_names:
        # ColumnTransformer often prefixes feature names (e.g., 'onehotencoder__job_admin.')
        # We need to remove these prefixes to match what XGBoost expects if it was trained on a DataFrame
        if '__' in name:
            cleaned_name = name.split('__', 1)[1] # Split only on the first '__'
            cleaned_names.append(cleaned_name)
        else:
            cleaned_names.append(name)
    return cleaned_names

# Load the trained churn prediction model
model = xgb.XGBClassifier() # Initialize an XGBClassifier
model.load_model("final_subscription_model.json") # Load the model from the native XGBoost format

# --- Define and fit the correct preprocessor directly in app.py ---
# Define numerical and categorical features
# Exclude 'pdays' and 'previous' based on the feature_names mismatch error
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign']
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Define the categories for OneHotEncoder based on frontend options (ensuring consistency)
job_categories = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital_categories = ['married', 'single', 'divorced', 'unknown']
education_categories = ['primary', 'secondary', 'tertiary', 'unknown']
default_categories = ['yes', 'no']
housing_categories = ['yes', 'no', 'unknown']
loan_categories = ['yes', 'no', 'unknown']
contact_categories = ['cellular', 'telephone', 'unknown']
month_categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcome_categories = ['failure', 'other', 'success', 'unknown']

all_categories = [
    job_categories, marital_categories, education_categories,
    default_categories, housing_categories, loan_categories,
    contact_categories, month_categories, poutcome_categories
]

# Create the preprocessing pipeline (ColumnTransformer)
# Use handle_unknown='ignore' for robust handling of unseen categories at inference
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=all_categories), categorical_cols)
    ],
    remainder='drop'
)

# Create a dummy DataFrame to fit the preprocessor at startup
# This ensures the scaler learns mean/std and encoder learns categories correctly.
# Using representative default values for each feature.
dummy_data = pd.DataFrame([{
    'age': 30, 'job': 'admin', 'marital': 'married', 'education': 'secondary',
    'default': 'no', 'balance': 1000, 'housing': 'yes', 'loan': 'no',
    'contact': 'cellular', 'day': 1, 'month': 'may', 'duration': 100,
    'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'
}])

# IMPORTANT: Adjust the dummy_data to match the numerical_cols and categorical_cols used in full_preprocessor.
# If 'pdays' and 'previous' are removed from numerical_cols, they should also be removed from dummy_data
# for consistency during fit.
# For now, we will leave them, as the preprocessor will drop them if not in numerical_cols.
# A cleaner approach would be to dynamically create dummy_data based on numerical_cols and categorical_cols.

# Fit the preprocessor on the dummy data once at application start
full_preprocessor.fit(dummy_data)

# NOTE: The original 'preprocessor.joblib' loading is commented out below
# because this new 'full_preprocessor' is taking over its intended role.
# If preprocessor.joblib was performing additional, necessary steps,
# it would need to be integrated carefully with this pipeline.
# try:
#     preprocessor = joblib.load("preprocessor.joblib")
# except FileNotFoundError:
#     print("Error: 'preprocessor.joblib' not found. Ensure your preprocessing pipeline is saved and available.")
#     exit()
# --- End of new preprocessor definition ---


# Define a route for the home page
@super_subscribe_api.get('/')
def home():
    return "Welcome to the Subscription Prediction API!" #code to define a welcome message

# Define an endpoint to predict churn for a single customer
@super_subscribe_api.post('/v1/predict')
def predict_subscriptions():
    # Get JSON data from the request
    data = request.get_json()

    # Convert the extracted raw data into a DataFrame
    raw_input_df = pd.DataFrame([{
        'age': data.get('age'),
        'job': data.get('job'),
        'marital': data.get('marital'),
        'education': data.get('education'),
        'default': data.get('default'),
        'balance': data.get('balance'),
        'housing': data.get('housing'),
        'loan': data.get('loan'),
        'contact': data.get('contact'),
        'day': data.get('day'),
        'month': data.get('month'),
        'duration': data.get('duration'),
        'campaign': data.get('campaign'),
        'pdays': data.get('pdays'), # Keep in raw_input_df for now, will be dropped by preprocessor
        'previous': data.get('previous'), # Keep in raw_input_df for now, will be dropped by preprocessor
        'poutcome': data.get('poutcome')
    }])

    # Define the exact column order expected by the preprocessor
    # This list must match the columns and their order used to fit the preprocessor
    # Note: 'pdays' and 'previous' are removed from numerical_cols, so they won't be processed by full_preprocessor.
    # However, they must be present in raw_input_df if the frontend sends them.
    expected_columns_for_preprocessor_input = [
        'age', 'job', 'marital', 'education', 'default', 'balance',
        'housing', 'loan', 'contact', 'day', 'month', 'duration',
        'campaign', 'pdays', 'previous', 'poutcome'
    ]

    # Ensure the raw_input_df has the correct columns in the correct order
    raw_input_df = raw_input_df[expected_columns_for_preprocessor_input]

    # Ensure 'day' column is numeric, as expected by StandardScaler
    if 'day' in raw_input_df.columns:
        # Attempt to convert to numeric, coercing errors. This will turn non-numeric values into NaN.
        raw_input_df['day'] = pd.to_numeric(raw_input_df['day'], errors='coerce')

        # Fill any NaNs that resulted from coercion (e.g., if a string became NaN).
        # Using 1 as a default for 'day'.
        raw_input_df['day'] = raw_input_df['day'].fillna(1).astype(int)

    # Apply the new 'full_preprocessor' to transform raw input
    processed_input_array = full_preprocessor.transform(raw_input_df)

    # Get feature names from the full_preprocessor
    feature_names = full_preprocessor.get_feature_names_out()

    # Clean up feature names to match what XGBoost expects (e.g., remove 'onehotencoder__' prefix)
    cleaned_feature_names = clean_feature_names(feature_names)

    # Create a DataFrame with the processed data and cleaned column names
    processed_input_data = pd.DataFrame(processed_input_array, columns=cleaned_feature_names)

    # --- NEW ADDITION: Reindex processed_input_data to match model's expected features ---
    # Get the feature names the model expects (from its training phase)
    expected_model_features = model.feature_names_in_

    # Reindex the processed_input_data to exactly match the model's expected features.
    # Fill any missing features (e.g., one-hot encoded categories not present in current input)
    # with 0. This is crucial for consistency.
    processed_input_data = processed_input_data.reindex(columns=expected_model_features, fill_value=0)
    # --- END NEW ADDITION ---

    # Make a churn prediction using the trained model
    prediction = model.predict(processed_input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Subscription': prediction})


# Run the Flask app in debug mode
if __name__ == '__main__':
    super_subscribe_api.run(debug=True)
