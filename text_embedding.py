import openai
import os
from openai import OpenAI
import streamlit as st
from openai import OpenAI
api_key = os.environ.get("OPENAI_API_KEY")

def get_embedding(text):

   client = OpenAI(api_key=api_key)
   text = text.replace("\n", " ")
   time.sleep(0.1)
   return client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding

import time

# Function to generate embeddings and update progress
def generate_embeddings_with_progress(df):
    embeddings = []  # List to collect embeddings
    progress_bar = st.progress(0)
    for i, row in df.iterrows():
        embedding = get_embedding(row['message'])
        # Ensure embedding is a list of floats; otherwise, handle the error
        embeddings.append(embedding)
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)

    df['embedding'] = embeddings
    return embeddings
    progress_bar.empty()

