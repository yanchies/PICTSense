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
            "content": f"Adopt a pessimistic point of view in the sentiment analysis.\
                You must only output a numerical sentiment score (example, 2), from a scale of 1 to 10 \
                with 1 being most negative and 10 being most positive, for the following survey response: {response}."
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
    results = []  # List to accumulate results

    def process_batch(batch):
        responses = batch['OER'].tolist()
        sentiments = analyze_sentiment_batch(responses)
        topics = identify_topic_batch(responses)
        return sentiments, topics

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch))

        for future in concurrent.futures.as_completed(futures):
            sentiments, topics = future.result()
            for j, sentiment in enumerate(sentiments):
                response_index = batch.index[j]  # Get original index for each response
                results.append({
                    "response_id": response_index,  # Original index as response_id
                    "response": batch.at[response_index, 'OER'],
                    "sentiment": sentiment,
                    "topic": topics[j] if j < len(topics) else "N/A"
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Ensure correct column order and reset index to avoid extra column
    results_df = results_df[['response_id', 'response', 'sentiment', 'topic']]
    results_df.reset_index(drop=True, inplace=True)  # Remove old index
    # Write results to JSON in one go
    with open(json_file_path, 'w', encoding='utf-8') as jsonf:
        json.dump(results_df.to_dict(orient='records'), jsonf, ensure_ascii=False, indent=4)

    print(f"JSON file saved to {json_file_path}")
    st.success("Successfully added new inputs!")
    return json_file_path
    # result = {}
    # # Process in batches
    # for i in range(0, len(df), batch_size):
    #     batch = df.iloc[i:i + batch_size]
    #     responses = batch['OER'].tolist()  

    #     sentiments = analyze_sentiment_batch(responses)
    #     topics = identify_topic_batch(responses)

    #     # Combine results for each response in the batch
    #     for j, row in enumerate(batch.itertuples(index=False)):
    #         result[f"response_{i + j + 1}"] = {
    #             "response": row.OER,
    #             "sentiment": sentiments[j] if j < len(sentiments) else "N/A",
    #             "topic": topics[j] if j < len(topics) else "N/A"
    #         }

    # # Final write to JSON file
    # with open(json_file_path, 'w', encoding='utf-8') as jsonf:
    #     json.dump(result, jsonf, ensure_ascii=False, indent=4)

    # print(f"JSON file saved to {json_file_path}")
    # st.success("Successfully added new inputs!")
    # return json_file_path

def get_df(json_file_path):
    df = pd.read_json(json_file_path)
    df = df.transpose().reset_index()
    return df