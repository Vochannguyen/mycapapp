import streamlit as st
import os
import pandas as pd
from PIL import Image
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from ast import literal_eval
import numpy as np
import openai
from sklearn.cluster import KMeans

# Ensure these modules are correctly implemented in your project directory
from clustering import kmeans_cluster
from text_embedding import get_embedding

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


from text_embedding import generate_embeddings_with_progress

import umap
from umap import UMAP

# Initialize OpenAI client
from openai import OpenAI
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def safe_convert_to_list(string):
    try:
        return literal_eval(string)
    except ValueError as e:
        print(f"Error converting string to list: {e}\nString: {string}")
        return []  # Return an empty list as a fallback
            

# Display an image
image = Image.open('micro.jpg')
st.image(image)
st.title("Generative AI Error Log Analyzer")
#st.header("Please upload your Error Log file here:")

# File uploader
data = st.file_uploader("Upload Error Log Files", type="csv")




if data is not None:
    df = pd.read_csv(data, encoding='latin1')
    
    st.write("**Please enter the number of clusters you wish to use for KMeans clustering:**")
    n_clusters = st.number_input('Number of Clusters', min_value=2, value=8, step=1)


    if st.button('Visualize Error Log Clusters'):
        embeddings = generate_embeddings_with_progress(df)

        # Convert string representations in 'embedding' column to actual lists of floats
        df['embedding'] = embeddings

        # Ensure embeddings are numpy arrays before clustering

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        # Perform UMAP dimensionality reduction
        reducer = UMAP()
        embeddings_2d = reducer.fit_transform(embeddings)

        # Update DataFrame with cluster labels
        df['Cluster'] = labels

        # Visualization with Plotly
        fig = px.scatter(
            x=embeddings_2d[:, 0], 
            y=embeddings_2d[:, 1], 
            color=df['Cluster'].astype(str),
            title="Cluster Visualization in 2D Space"
        )
 
        st.plotly_chart(fig)

    if st.button('Analyze Log Errors'):
                
                embeddings = generate_embeddings_with_progress(df)
                matrix = np.stack(df["embedding"].values)

                

                n_clusters = 9

                kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
                kmeans.fit(matrix)

                labels = kmeans.labels_
                df["Cluster"] = labels.astype(str)
                df['Cluster'] = df['Cluster'].astype(int)
                # Convert labels to string type
                logs_per_cluster = 10

                for i in range(n_clusters):
                    st.text(f"Cluster {i} Anomalies:")

                    # Filter the DataFrame for the current cluster
                    cluster_df = df[df["Cluster"] == i]

                    # Check if there are enough logs in the cluster to sample from
                    if len(cluster_df) >= logs_per_cluster:
                        # Extract a sample of error logs from each cluster
                        error_logs = "\n".join(
                            cluster_df["message"]
                            .sample(n=logs_per_cluster, random_state=42)
                            .values
                        )
                    else:
                        # If not enough logs, use whatever is available
                        error_logs = "\n".join(cluster_df["message"].values)
                    
                    # Create a prompt for GPT-4-5 turbo to analyze the error logs
                    response = client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        #Fine-Tune the Message Prompt
                        messages=[
                            {"role": "system", "content": "You are providing data analysis on Microsoft Azure Service Bus error log messages. This cloud service is designed to facilitate communication between applications and services. In analyzing Azure Service Bus error logs, common issues fall into categories such as permissions, connectivity, sender, receiver, processor, and transaction-related problems. These encompass handling Service Bus exceptions, managing access rights, addressing network connection challenges, troubleshooting message sending and receiving difficulties, resolving processor operation concerns, and navigating transaction complexities. Your primary role is identifying log error patterns clustered around a suspected anomaly and summarizing what the clusters of log messages represent."},
                            {"role": "user", "content": f'Error Logs:\n"""\n{error_logs}\n"""\n\nIdentified Patterns or Anomalies:'}
                        ],
                        temperature=0.1,
                        max_tokens=300,  # Adjusted max_tokens for potentially more detailed analysis
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    # Display GPT response
                    st.text(response.choices[0].message.content.strip())

                    # Optionally, display additional information from the sampled logs
                    # Check again if there are enough logs
                    if len(cluster_df) >= logs_per_cluster:
                        sample_cluster_logs = cluster_df.sample(n=logs_per_cluster, random_state=42)
                    else:
                        sample_cluster_logs = cluster_df

                    for j in range(len(sample_cluster_logs)):
                        st.text(sample_cluster_logs.iloc[j].get('message', '')[:70])  # Display the first part of the error log
 



            # Number of error logs to review per cluster
            














    # Text area for summarization
    #st.subheader("Exploratory Data Analysis")
    #summarization_result = st.text_area("Visualization:", "")

    # Text area for summarization
    #st.subheader("Clustering")
    #summarization_result = st.text_area("Unsupervised Learning Model:", "")

    #st.subheader("AI Summarization of Error Logs")
    #summarization_result = st.text_area("Summarization:", "")

    #st.subheader("AskErrorLogs")
    #query = st.text_area("Enter Query:")
    #button = st.button("Generate Response:")