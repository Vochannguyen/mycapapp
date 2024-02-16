import pandas as pd
from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent  # Assuming this is the correct import

def query_agent(data, query):

    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    llm = OpenAI()
    
    # Create a Pandas DataFrame agent.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

 
    return agent.run(query)


