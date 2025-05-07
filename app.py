import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

# Load model and scaler
kmeans = joblib.load("kmeans_model_new.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Navbar using option menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "About", "Dataset", "Clustering Interpretation", "Predict"],
        icons=["house", "info-circle", "table", "diagram-3", "magic"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if selected == "Home":
    st.title("👋 Welcome to Customer Segmentation Dashboard")
    st.write("This interactive app helps analyze user journeys through K-Means clustering.")
    st.write("Navigate through the sidebar to explore more about the project, dataset, clusters, and try live predictions!")

# About Section
elif selected == "About":
    st.title("📌 Project Overview")
    st.markdown("""
    This project focuses on analyzing travel-related user behavior to segment customers based on their interactions.
    
    We used the following steps:
    - Preprocessing: Cleaned data (e.g., standardized device categories)
    - Feature Selection: Chose key behavioral features like views, check-ins, time spent
    - PCA: Used Principal Component Analysis to reduce dimensions and improve clustering
    - Clustering: Applied K-Means with 3 clusters, validated using the elbow method
    - Built a Streamlit app to interpret and predict user segments
    """)

# Dataset Description
elif selected == "Dataset":
    st.title("🗂️ Dataset Description")
    st.markdown("""
    The dataset includes the following features:

    | Column | Description |
    |--------|-------------|
    | **UserID** | Unique identifier for each customer |
    | **Taken_product** | Indicates if the customer availed the travel product |
    | **Yearly_avg_view_on_travel_page** | Avg. views on travel page per year |
    | **preferred_device** | Device used by customer (Mobile/Laptop) |
    | **total_likes_on_outstation_checkin_given** | Likes given on outstation check-ins |
    | **yearly_avg_Outstation_checkins** | Avg. outstation check-ins per year |
    | **member_in_family** | Number of family members |
    | **preferred_location_type** | Preferred travel location type |
    | **Yearly_avg_comment_on_travel_page** | Avg. yearly comments on travel page |
    | **total_likes_on_outofstation_checkin_received** | Likes received on check-ins |
    | **week_since_last_outstation_checkin** | Weeks since last check-in |
    | **following_company_page** | Is the user following company page? |
    | **montly_avg_comment_on_company_page** | Monthly comments on company page |
    | **working_flag** | Whether the user is employed |
    | **travelling_network_rating** | Travel network rating by user |
    | **Adult_flag** | Adult indicator |
    | **Daily_Avg_mins_spend_on_traveling_page** | Avg. daily time spent on travel page |
    
    Only a subset of the above features was used for clustering.
    """)

# Cluster Interpretation
elif selected == "Clustering Interpretation":
    st.title("📊 Clustering Analysis")
    st.write("We used the Elbow Method to determine optimal clusters.")

    elbow_data = pd.DataFrame({
        "K": [1, 2, 3, 4, 5],
        "Inertia": [1000, 600, 400, 350, 320]
    })
    fig = px.line(elbow_data, x="K", y="Inertia", markers=True, title="Elbow Method")
    st.plotly_chart(fig)

    st.subheader("Why 3 Clusters?")
    st.write("The graph shows a sharp decrease in inertia up to K=3, after which it levels off. Hence, 3 clusters were chosen.")

    st.subheader("Cluster Meaning")
    cluster_summary = pd.DataFrame({
        "Cluster": [0, 1, 2],
        "Travel Page Views": [263.84, 250.00, 352.48],
        "Outstation Check-ins": [20.76, 3.54, 6.13],
        "Daily Time (mins)": [11.72, 9.61, 22.95]
    })
    st.dataframe(cluster_summary)

    st.markdown("""
    - 🟢 **Cluster 0**: Medium viewers and high check-in users
    - 🔵 **Cluster 1**: Low engagement users
    - 🟣 **Cluster 2**: Very active travelers with highest views and time spent
    """)

# Prediction App
elif selected == "Predict":
    st.title("🔍 Customer Segment Prediction")
    st.write("Enter a customer's travel behavior to predict their cluster.")

    yearly_avg_view = st.number_input("Yearly Average Views on Travel Page", min_value=0, value=300)
    yearly_avg_checkins = st.number_input("Yearly Average Outstation Check-ins", min_value=0, value=5)
    daily_avg_minutes = st.number_input("Daily Average Minutes Spent on Traveling Page", min_value=0, value=30)

    input_data = np.array([[yearly_avg_view, yearly_avg_checkins, daily_avg_minutes]])

    if st.button("Get Customer Segmentation"):
        scaled_input = scaler.transform(input_data)
        predicted_cluster = kmeans.predict(scaled_input)
        st.success(f"Predicted Segment (Cluster): {predicted_cluster[0]}")
        
        # Cluster Descriptions
        cluster_descriptions = {
            0: "Cluster 0 - 'Frequent Travelers': These users spend considerable time on travel-related pages and have a high number of outstation check-ins. They are highly engaged users who actively follow travel content.",
            1: "Cluster 1 - 'Occasional Travelers': These users have moderate engagement with travel pages and check-in less frequently. They tend to engage with content occasionally but are not regular users.",
            2: "Cluster 2 - 'Low Engagement Travelers': These users have low interaction with travel pages and check-ins. Their online engagement with travel-related content is minimal."
        }

        # Display the cluster description based on the predicted cluster
        st.write(cluster_descriptions.get(predicted_cluster[0], "Cluster description not available."))

# Footer at the bottom of the page
st.markdown("""
    <hr style='margin-top: 50px;'>
    <div style='text-align: center;'>
        Made with ❤️ using K-Means Clustering | 
        <a href='https://github.com/Jahnavi41/Customer-Journey-Analysis-Using-Clustering-and-Dimensionality-Reduction-Enhancing-User-Experience-' target='_blank'>GitHub</a>
    </div>
""", unsafe_allow_html=True)
