
import streamlit as st
import requests

st.set_page_config(layout='wide')

# Create a row for the header and button
col_header_title, col_header_button = st.columns([3, 1]) # 3:1 ratio for title vs button

with col_header_title:
    st.markdown("<h1 style='text-align: left; color: #4CAF50;'>Bank Marketing Optimisation 📈</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; color: #2196F3;'>Customer Subscription Prediction ✨</h2>", unsafe_allow_html=True)

with col_header_button:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True) # Spacer for vertical alignment
    predict_button_pressed = st.button("Predict Subscription", type='primary')


# Input fields for the features expected by the backend API
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week_options = ['mon', 'tue', 'wed', 'thu', 'fri']
job_options = ['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
housing_options = ['yes', 'no', 'unknown']
loan_options = ['yes', 'no', 'unknown']
marital_options = ['married', 'single', 'divorced', 'unknown']
education_options = ['primary', 'secondary', 'tertiary', 'unknown']
default_options = ['yes', 'no']
contact_options = ['cellular', 'telephone', 'unknown']
poutcome_options = ['failure', 'other', 'success', 'unknown']

# Create two main columns for a more organized layout of inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='color: #2196F3;'>Personal & Financial Details 💼</h3>", unsafe_allow_html=True)
    # Nested columns to reduce input field size
    nested_col1_1, nested_col1_2 = st.columns(2)
    with nested_col1_1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox("Marital Status", marital_options)
        default = st.selectbox("Has Credit in Default?", default_options)
        loan = st.selectbox("Has Personal Loan?", loan_options)
    with nested_col1_2:
        job = st.selectbox("Job Type", job_options)
        education = st.selectbox("Education Level", education_options)
        balance = st.number_input("Average Yearly Balance (in Euros)", value=1000)
        housing = st.selectbox("Has Housing Loan?", housing_options)


with col2:
    st.markdown("<h3 style='color: #2196F3;'>Campaign & Contact Information 📞</h3>", unsafe_allow_html=True)
    # Nested columns to reduce input field size
    nested_col2_1, nested_col2_2 = st.columns(2)
    with nested_col2_1:
        contact = st.selectbox("Contact Communication Type", contact_options)
        day_of_week_str = st.selectbox("Day of Week of Last Contact", day_of_week_options)
        campaign = st.number_input("Number of Contacts During Campaign", min_value=1, value=1)
        pdays = st.number_input("Days Since Previous Contact", min_value=-1, value=-1)
    with nested_col2_2:
        month = st.selectbox("Month of Last Contact", month_options)
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, value=100)
        previous = st.number_input("Previous Campaign Contacts", min_value=0, value=0)
        poutcome = st.selectbox("Outcome of Previous Campaign", poutcome_options)


# Map day of week string to numerical value (1-5 for mon-fri)
day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
day_numerical = day_mapping.get(day_of_week_str, 1) # Default to 1 if not found

# Prepare data for API request (ensure keys match backend's expected keys)
prediction_data = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day_numerical, # Sending numerical day to backend
    "month": month, # Sending as Month to backend
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

if predict_button_pressed:
    # IMPORTANT: Replace <user_name> and <space_name> with your Hugging Face username and backend space name
    # The repo_id for the backend is dcsamuel/SuperMarketing
    api_endpoint = "https://dcsamuel-SuperMarketing.hf.space/v1/predict"

    response = requests.post(api_endpoint, json=prediction_data)

    if response.status_code == 200:
        result = response.json()
        prediction = result["Subscription"]

        if prediction == 1:
            st.success("**Prediction: The customer will likely subscribe! 🎉**")
        else:
            st.info("**Prediction: The customer will likely NOT subscribe. 😔**")

    else:
        st.error(f"Error in API request: {response.status_code} - {response.text}")
        st.write("Please ensure the backend API is running and accessible.")
