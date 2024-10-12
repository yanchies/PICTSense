import json
import pandas as pd
import streamlit as st
from helper_functions import llm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# Assuming OpenAI has been initialized and imported
client = llm.client

def analyze_sentiment_batch(responses):
    results = []
    
    for response in responses:
        # Prepare a single message for each response
        message = {
            "role": "user", 
            "content": f"Only provide a numerical sentiment score on for the following survey response using\
                  a scale of 1 to 10 with 1 being most negative and 10 being most positive: {response}. \
                  Adopt a pessimistic tone in the analysis as the responses are typically succinctly negative."
        }
        
        # Call the OpenAI API for each response
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[message],
                max_tokens=60
            )
            sentiment = response.choices[0].message.content.strip()
            results.append(sentiment)
        except Exception as e:
            print(f"Error analyzing sentiment for response: {response}. Error: {e}")
            results.append("Error")
    
    return results

def get_embeddings(responses):
    """This function retrieves embeddings for the given responses using the OpenAI API."""
    embeddings = []
    
    # Prepare the batch requests for embeddings
    for response in responses:
        try:
            # Call the OpenAI API to get the embedding
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",  # Adjust model as needed
                input=response
            )
            embedding = embedding_response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error getting embeddings for response: {response}. Error: {e}")
            embeddings.append(None)  # Append None or a placeholder for error handling
            
    return embeddings

# use embeddings approach to enhance accuracy
def identify_topic_batch(responses):
    results = []
    
    categories = [
        'Equipment Serviceability',
        'Lodging & Food',
        'Training Effectiveness',
        'Administration & Time Management',
        'Leadership',
        'Health & Safety',
        'Training Experience',
        'Comaraderie & Morale',
        'Inconclusive'
    ]
    
    category_embeddings = get_embeddings(categories)
    response_embeddings = get_embeddings(responses)

    # Iterate through the responses and their embeddings
    for i, response in enumerate(responses):
        current_embedding = response_embeddings[i]
        
        # Handle cases where embedding retrieval might fail
        if current_embedding is None:
            results.append("Error")
            continue

        similarities = cosine_similarity([current_embedding], category_embeddings)
        max_index = np.argmax(similarities) # Get the topic category with the highest similarity
        topic = categories[max_index]  # Map index to topic category

        results.append(topic)
    
    return results

def process_responses(df, json_file_path, batch_size=100):
    result = {}
    def process_batch(batch):
        responses = batch['OER'].tolist()
        sentiments = analyze_sentiment_batch(responses)
        topics = identify_topic_batch(responses)
        return sentiments, topics

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, df.iloc[i:i + batch_size]) for i in range(0, len(df), batch_size)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            sentiments, topics = future.result()
            batch = df.iloc[i * batch_size:(i + 1) * batch_size]
            for j, row in enumerate(batch.itertuples(index=False)):
                result[f"response_{i * batch_size + j + 1}"] = {
                    "response": row.OER,
                    "sentiment": sentiments[j] if j < len(sentiments) else "N/A",
                    "topic": topics[j] if j < len(topics) else "N/A"
                }

    with open(json_file_path, 'w', encoding='utf-8') as jsonf:
        json.dump(result, jsonf, ensure_ascii=False, indent=4)

    print(f"JSON file saved to {json_file_path}")
    st.success("Successfully added new inputs!")
    return json_file_path

def get_df(json_file_path):
    df = pd.read_json(json_file_path)
    df = df.transpose().reset_index()
    return df