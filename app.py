import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pandas as pd  # Import pandas for reading CSV
from utils import query_agent

load_dotenv()

image = Image.open('micro.jpg')

st.image(image)
st.title("Generative AI Error Log Analyzer")
st.header("Please upload your Error Log file here:")

# Capture the CSV file
data = st.file_uploader("Upload Error Log Files", type="csv")

# Check if a CSV file is uploaded
if data is not None:
    # Read the CSV file with a specified encoding (e.g., 'latin1' for ISO-8859-1 encoding)
    df = pd.read_csv(data, encoding='latin1')

    # Text area for summarization
    st.subheader("Exploratory Data Analysis")
    summarization_result = st.text_area("Visualization:", "")

    # Text area for summarization
    st.subheader("Clustering")
    summarization_result = st.text_area("Unsupervised Learning Model:", "")

    st.subheader("AI Summarization of Error Logs")
    summarization_result = st.text_area("Summarization:", "")

    st.subheader("AskErrorLogs")
    query = st.text_area("Enter Query:")
    button = st.button("Generate Response:")

    if button:
        # Get Response
        answer = query_agent(df, query)  # Pass the DataFrame instead of the file
        st.write(answer)
