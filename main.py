import pandas as pd
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    # print(response)
    return response


df = joblib.load('embedding.joblib')
incoming_query = input("Ask a Question :")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [
                                 question_embedding]).flatten()
top_result = 5
max_idx = similarities.argsort()[::-1][0:top_result]

new_Df = df.loc[max_idx]
# print(new_Df[['title','number','Text']])
context = "\n".join(
    f"[{row.Start:.1f}s - {row.End:.1f}s] {row.Text}"
    for _, row in new_Df.iterrows()
)

prompt = f"""
User Question:
{incoming_query}

----------------------------------------
Transcript Context:
{context}

----------------------------------------
Instructions:
- Answer only from the context.
- Be precise and remove redundancy.
"""

# for index, item in new_Df.iterrows() :
#     print(index, item['title'], item['Text'], item['Start'],item['End'])
with open("prompt.txt", "w") as f:
    f.write(prompt)
response = inference(prompt)['response']

with open('response.txt', 'w') as f:
    f.write(response)
