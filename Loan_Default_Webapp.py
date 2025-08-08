import streamlit as st
import pandas as pd
import numpy as np
import time

# Reboot button at the TOP of the sidebar
if st.sidebar.button("Reboot App 🔄"):
    st.session_state.clear()
    st.experimental_rerun()

# Existing sidebar content (unchanged)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Settings"])

st.sidebar.header("Filters")
filter_date = st.sidebar.date_input("Select date")
filter_category = st.sidebar.selectbox("Category", ["All", "A", "B", "C"])

# Main content area
st.title(f"{page} Page")
st.write(f"Current filters: {filter_date}, {filter_category}")

# Page-specific content
if page == "Dashboard":
    st.subheader("Performance Metrics")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(chart_data)
    
    with st.expander("Detailed Stats"):
        st.dataframe(chart_data.describe())
        
elif page == "Analytics":
    st.subheader("Data Analysis")
    n = st.slider("Data points", 10, 100, 50)
    arr = np.random.normal(1, 1, size=n)
    st.histogram(arr, bins=20)
    
elif page == "Settings":
    st.subheader("Configuration")
    api_key = st.text_input("API Key", type="password")
    save_config = st.checkbox("Save settings")
    
    if st.button("Apply Settings"):
        with st.spinner("Saving..."):
            time.sleep(1.5)
            st.success("Settings saved!")
            st.balloons()

# Footer (unchanged)
st.sidebar.markdown("---")
st.sidebar.caption("© 2023 My Streamlit App | v1.2.0")