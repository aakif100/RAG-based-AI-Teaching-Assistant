
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
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()['response']
    return response

df = joblib.load("joblib/embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df["embedding"]) , [question_embedding]).flatten()
# print(similarities)

top_matches = (-similarities).argsort()[0:6]
# print("Top Matches Chunk IDs: ", top_matches)




new_df = df.loc[top_matches]
# print(new_df["number"])   

prompt = f''' 
here are the video chunks containing video title , video number ,start time in seconds and end time in seconds and text at that time:
"{new_df[["title" , "number" , "start" , "end" , "text"]].to_json(orient="records")}" 
answer the question based on the video chunks provided only also give an explanation at the end about the concept of the question asked.


----------------------------------------------------------------
"{incoming_query}"
from the above income query the user has asked about questions regarding the videos of the course which i am doing ok.. so this course is called sigma web development course and the user will ask questions based on the videos of the cpurse only and you will have to answer them based on the video chunks provided above only and you cannot answer unrelated questions ok got it. answer to the user with all the datas you have got and guide them to the video and timestamps where that topic is covered ok got it. if the user asks unrelated questions then you will have to tell them that you can only answer related questions to the video chunks provided and you cannot answer unrelated questions ok got it.
        '''

with open("prompt.txt" , "w" , encoding="utf-8") as f:
    f.write(prompt)


model_response = (inference(prompt))
print(model_response)

with open("response.txt" , "w" , encoding="utf-8") as f:
    f.write(model_response)