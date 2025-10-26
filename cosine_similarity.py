
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

df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df["embedding"]) , [question_embedding]).flatten()
# print(similarities)

top_matches = (-similarities).argsort()[0:10]
# print("Top Matches Chunk IDs: ", top_matches)




new_df = df.loc[top_matches]
# print(new_df[["title" , "text" , "number"]])   



# for index , item in new_df.iterrows():
#     print( item['number']  , item['title'] , item['text'] , item['start'] , item['end'] )
print(df[0:5])
# for item in new_df:
#     print(item['title'] ,  item['number'] , item['text'])
