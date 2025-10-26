# checking with a sample request to the embedding server

# import requests

# r = requests.post('http://localhost:11434/api/embeddings' , json={
#     "model": "bge-m3",
#     "prompt": "A photo of a cat"})

# embedding =r.json()['embedding']

# print(embedding[0:5]) 



# the above post-request path proved it is slowing down the process so moving to batch embedding creation below..
import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("jsons")  
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    # print(embeddings)
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        # if i == 5:
        #     break 
    
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df , "embeddings.joblib")
# print(df)

