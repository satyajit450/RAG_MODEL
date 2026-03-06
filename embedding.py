from ollama import embed
def embedding(text):
    response = embed(
        model='bge-m3',
        input=text
    )
    return response

print(embedding("Hello, world!")["embeddings"][0][:10])