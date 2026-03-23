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
    print(response)
    return response


df = joblib.load('embedding.joblib')
incoming_query = input("Ask a Question :")
question_embedding = create_embedding([incoming_query])[0]

similarities =  cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
top_result = 5
max_idx = similarities.argsort()[::-1][0:top_result]

new_Df = df.loc[max_idx]
# print(new_Df[['title','number','Text']])
prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_Df[["title", "number", "Start", "End", "Text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
# for index, item in new_Df.iterrows() :
#     print(index, item['title'], item['Text'], item['Start'],item['End'])
with open("prompt.txt", "w") as f :
    f.write(prompt)