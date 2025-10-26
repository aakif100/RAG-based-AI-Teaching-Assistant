
# import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from read_chunks import create_embedding
# reflect why i deleted the above line because it was embedding think about it
import joblib
import requests




def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-r1",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()["text"]
    return response

# slacked this model because it is very slow bruhhhh , so going for llamas faster model (as of the time im writing this code) is llama3.2


df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df["embedding"]) , [question_embedding]).flatten()
# print(similarities)

top_matches = (-similarities).argsort()[0:6]
# print("Top Matches Chunk IDs: ", top_matches)




new_df = df.loc[top_matches]
# print(new_df[["title" , "text" , "number"]])   

prompt = f''' I am teaching web development called sigma web development course and here are video chunks containing video title , video number ,start time in seconds and end time in seconds and text at that time:
"{new_df[["title" , "number" , "start" , "end" , "text"]].to_json()}" # made in json format from pandas so that llm can understand easily



----------------------------------------------------------------
"{incoming_query}"
User asked the question related to the video chunks  , you have to answer where and how much content is taught in which video (in which video and what timestamps) and guide user to that part of the video for better understanding. If user asks unrelated questions  , tell him  that you can only answer the related questions to the video chunks provided and you cannot answer unrelated questions.
        '''

with open("prompt.txt" , "w" , encoding="utf-8") as f:
    f.write(prompt)


print(inference(prompt))