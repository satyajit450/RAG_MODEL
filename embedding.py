from ollama import embed
import os, json
import pandas as pd

def create_embedding(text):
    response = embed(
        model='bge-m3',
        input=text
    )
    return response

jsons = os.listdir("chunks")
my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f"chunks/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")

    for chunk in content['chunks']:  # simple loop, no enumerate needed
        chunk['chunk_id'] = chunk_id
        chunk_id += 1

        # create embedding per chunk
        emb = create_embedding(chunk["Text"])["embeddings"][0]
        chunk["embedding"] = emb

        my_dict.append(chunk) 

# create dataframe
df = pd.DataFrame.from_records(my_dict)
# print only summary, not full vectors
# print(df)