import requests
import os
import numpy as np
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

jsons = os.listdir("chunks")  # or "jsons" (choose one folder)

my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"chunks/{json_file}",encoding='utf-8') as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    embeddings = create_embedding([
        c.get('text') or c.get('Text') for c in content['chunks']
    ])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        if(i==5):
            break
    break

df = pd.DataFrame.from_records(my_dicts)

# incoming_query = input("Ask a Question :")
# question_embedding = create_embedding([incoming_query])[0]

# similarities =  cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# top_result = 3
# max_idx = similarities.argsort()[::-1][0:top_result]

# new_Df = df.loc[max_idx]
# print(new_Df[['title','number','Text']])
print(df)