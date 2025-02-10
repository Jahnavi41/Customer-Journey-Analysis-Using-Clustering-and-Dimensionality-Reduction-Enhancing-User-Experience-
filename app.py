import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained K-Means model
kmeans = joblib.load('kmeans_model.pkl')  # Path to your saved K-Means model
scaler = StandardScaler()

# Set the title of the app
st.title('Customer Segmentation Using K-Means Clustering')

# Input fields for the features used during training
st.header('Enter Customer Data')

# Correcting the input fields (only 3 features, matching the training data)
yearly_avg_view = st.number_input('Yearly Average Views on Travel Page', min_value=0, value=300)
yearly_avg_checkins = st.number_input('Yearly Average Outstation Check-ins', min_value=0, value=5)
daily_avg_minutes = st.number_input('Daily Average Minutes Spent on Traveling Page', min_value=0, value=30)

# Collecting the input data into an array for prediction
input_data = np.array([[yearly_avg_view, yearly_avg_checkins, daily_avg_minutes]])

# Button to trigger prediction
if st.button('Get Customer Segmentation'):
    # Standardize the input data using the scaler
    scaled_input = scaler.fit_transform(input_data)
    
    # Make predictions using the K-Means model
    predicted_cluster = kmeans.predict(scaled_input)
    
    # Display the predicted segment (cluster)
    st.write(f"Predicted Segment (Cluster): {predicted_cluster[0]}")
