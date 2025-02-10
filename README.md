# Customer-Journey-Analysis-Using-Clustering-and-Dimensionality-Reduction-Enhancing-User-Experience

## Overview

This project focuses on segmenting customers based on their journey and behavior data using **Clustering** and **Dimensionality Reduction** techniques. The goal is to group customers into distinct segments for better targeting, personalized experiences, and insights into their preferences and behavior.

### Features of the Project:
1. **Customer Segmentation**: Using **K-Means Clustering** to group customers based on their data, including average views, check-ins, and time spent on travel-related pages.
2. **Dimensionality Reduction**: Using techniques like **PCA (Principal Component Analysis)** to reduce the dimensionality of the data, making it more manageable while preserving key information.
3. **Web Application**: A user-friendly **Streamlit** app allows users to input customer data, view predictions, and interact with the segmentation model in real time.

---

## Project Description

### Problem Statement:
The goal is to classify customers into distinct groups based on their behavior data. By clustering customers, we can provide personalized experiences and insights into customer behavior patterns.

### Data Used:
The dataset used for this analysis contains several customer attributes such as:
- **Yearly Average Views on Travel Page**
- **Yearly Average Outstation Check-ins**
- **Daily Average Minutes Spent on Traveling Page**

### Approach:
1. **Data Preprocessing**:
   - Handle missing values.
   - Normalize the data to ensure uniformity for the clustering algorithm.
2. **Clustering**:
   - Use **K-Means** to cluster customers into distinct groups.
   - Apply **Dimensionality Reduction** using **PCA** to visualize the clusters in a lower-dimensional space.
3. **Model Deployment**:
   - Deploy the trained model and interface it with a **Streamlit** web app, which provides an interactive experience for users.

---

## Deployed Application

The deployed application allows users to enter customer attributes and see the predicted cluster (segment) for the entered data. You can access the application at the following link:

**[Customer Journey Segmentation Web App](https://jeucnqwx6ndhepa9rs4mkbe.streamlit.app/)**

### How it works:
1. Enter customer data such as:
   - Yearly Average Views on Travel Page
   - Yearly Average Outstation Check-ins
   - Daily Average Minutes Spent on Traveling Page
2. Click **"Get Customer Segmentation"**.
3. The app will return the predicted customer segment based on the provided data.
