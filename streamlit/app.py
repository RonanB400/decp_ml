import streamlit as st

# Sidebar: Module selector
module = st.sidebar.radio("Choose a module:", ["Similar Markets", "Amount Estimation", "Anomaly Detection"])

# Main Panel
st.title("Business Intelligence App")

if module == "Similar Markets":
    st.header("📍 Similar Markets")
    # Add inputs, visuals, etc. for similar markets
    st.write("Display clustering or similarity visualization here.")

elif module == "Amount Estimation":
    st.header("💰 Amount Estimation")
    # Add model prediction input/output here
    st.write("Add feature inputs and predicted amount display here.")

elif module == "Anomaly Detection":
    st.header("🚨 Anomaly Detection")
    # Add anomaly detection inputs/output here
    st.write("Upload file or select row to flag anomalies.")
