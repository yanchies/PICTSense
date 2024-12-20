import json
import pandas as pd
import streamlit as st
from helper_functions import llm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import altair as alt

# """  
# This file contains the relevant LLM-based and streamlit functions.   
# """ 

client = llm.client

def analyze_sentiment_batch(responses):
    results = []
    
    for response in responses:
        # Prepare a single message for each response
        message = {
            "role": "user", 
            "content": f"You are a sentiment analyst who works with a numerical scale of 1 to 10 to give sentiment \
                scores. A score of 1 refers to a response which exhibits extreme dissatisfaction and negativity \
                    from language, content, and tone. 10 refers to extreme positivity and satisfaction. You must \
                        ONLY output a numerical sentiment score (example, 2) for the following survey response: {response}."
        }
        
        # Call the OpenAI API for each response
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[message],
                max_tokens=60
            )
            sentiment = response.choices[0].message.content.strip()
            results.append(sentiment)
        except Exception as e:
            print(f"Error analyzing sentiment for response: {response}. Error: {e}")
            results.append(0)
    
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

# use embeddings approach to enhance accuracy for topic identification
def identify_topic_batch(responses):
    results = []
    
    categories = [
        'Equipment Serviceability',
        'Lodging & Food',
        'Training Effectiveness',
        'Administration',
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
    results = []  

    def process_batch(batch, start_index):
        responses = batch['OER'].tolist()
        sentiments = analyze_sentiment_batch(responses)
        topics = identify_topic_batch(responses)

        # Collect results for the current batch with correct indices
        batch_results = []
        for j, sentiment in enumerate(sentiments):
            response_index = start_index + j  
            batch_results.append({
                "response_id": str(response_index + 1),
                "response": df.loc[response_index, 'OER'],
                "sentiment": sentiment,
                "topic": topics[j] if j < len(topics) else "N/A"
            })
        return batch_results

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch, i))  

        # Collect results from all futures
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())  # Extend with batch results

    # Write results to JSON in one go
    with open(json_file_path, 'w', encoding='utf-8') as jsonf:
        json.dump(results, jsonf, ensure_ascii=False, indent=4)

    print(f"JSON file saved to {json_file_path}")
    st.success("Successfully added new inputs!")
    return json_file_path


def visualise(df):
    sentiment_bar = df["sentiment"].value_counts().sort_index()
    topic_bar = df["topic"].value_counts()

    neg = df[df['sentiment'].between(1, 4)]['topic'].value_counts()
    neg_df = neg.reset_index()

    pos = df[df['sentiment'].between(6, 10)]['topic'].value_counts()
    pos_df = pos.reset_index()

    # metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("No. of Responses", df.shape[0])
    col2.metric("Negative Responses (1 to 4)", neg.sum())
    col3.metric("Neutral Responses (5)", df[df['sentiment'] == 5].shape[0])
    col4.metric("Positive Responses (6 to 10)", pos.sum())
    
    st.divider()
    # visualise sentiment scores and topics
    col1, col2 = st.columns(2)
    with col1:
        st.write("Sentiment Scores:")
        st.bar_chart(data=sentiment_bar, x_label= "Sentiment Score", y_label="Count")
    with col2:
        st.write("Topics:")
        topic_df = topic_bar.reset_index()
        topic_chart = (alt.Chart(topic_df).mark_bar(color='#7c7c7c').encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("topic:N", sort=None, title="Topic", axis=alt.Axis(labelLimit=200))
        ))
        st.altair_chart(topic_chart, use_container_width=True)


    st.divider()
    # show the top negative topics
    st.write("Top Negative Topics:")
    neg_chart = (alt.Chart(neg_df).mark_bar(color='#d9061b').encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("topic:N", sort=None, title="Topic", axis=alt.Axis(labelLimit=200))
    ))
    st.altair_chart(neg_chart, use_container_width=True)

    st.write("Top Positive Topics:")
    # show the top positive topics
    pos_chart = (alt.Chart(pos_df).mark_bar(color='#e7e7df').encode(
        x=alt.X("count:Q", title="Count", axis=alt.Axis(format='d')),
        y=alt.Y("topic:N", sort=None, title="Topic", axis=alt.Axis(labelLimit=200))
    ))
    st.altair_chart(pos_chart, use_container_width=True)

    st.divider()
    # pivot table
    st.subheader("Summary of Sentiments and Responses by Topic")
    pivot_df = df.groupby("topic").agg(
    avg_sentiment=("sentiment", "mean"),
    count=("sentiment", "count"),
    response_summary=("response", lambda x: " | ".join(x))).reset_index()
    pivot_df['response_summary'] = pivot_df['response_summary'].apply(summarize)
    st.table(pivot_df.sort_values(by="avg_sentiment", ascending=True))


# Function to generate summary using OpenAI LLM
def summarize(response_summary):
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant summarizing feedback in a concise manner.\
              Avoid mentioning specific feedback but try to capture most of the feedbacks."},
            {"role": "user", "content": f"Summarize the following responses in a concise manner: {response_summary[:500]}"}
        ],
        max_tokens=100,
        temperature=0.5,  # Lower temperature for concise summaries
    )
    
    summary = response.choices[0].message.content.strip()
    return summary
    
def gen_df():
    responses = []
    categories = [
        'Equipment Serviceability',
        'Lodging & Food',
        'Training Effectiveness',
        'Administration',
        'Leadership',
        'Health & Safety',
        'Training Experience',
        'Comaraderie & Morale',
    ]
    
    for i in range(100):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You have just gone through a 2-week \
                     reservist military training. You might be unhappy about your experience or satisfied with it."},
                    {"role": "user", "content": f"Generate a survey feedback on the in-camp training, \
                     based the following type of issue: {categories[i%8]}. Do not generate headers."}
                ],
                max_tokens=50,
                temperature=1  # Adjust creativity as needed
            )
            
            response_text = response.choices[0].message.content.strip()
            response_id = f"response_{i+1}"
            responses.append({
                "response_id": response_id,
                "OER": response_text
            })
        
        except Exception as e:
            print(f"Error generating response {i+1}: {e}")
    
    # Convert responses to a DataFrame
    response_df = pd.DataFrame(responses)
    return response_df

