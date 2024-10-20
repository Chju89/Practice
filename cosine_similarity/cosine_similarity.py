import gensim.downloader as api
import numpy as np

###Cosine similarity between vector1 and vector2 with model.similarity()
model = api.load("glove-twitter-200")
word1 = "marriage"
word2 = "happiness"
vector1 = model[word1]
vector2 = model[word2]
result = model.similarity(word1, word2)
print(f"Cosine similarity between vector1 and vector2 using model.similarity(): {result}")

###Cosine similarity between vector1 and vector2 without numpy
def cosine_similarity_without_numpy(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    norm_a = norm(vector1)
    norm_b = norm(vector2)
    similarity = dot_prod / (norm_a * norm_b)
    return similarity
def dot_product(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))
def norm(vector):
    return sum(x ** 2 for x in vector) ** 0.5
similarity_score = cosine_similarity_without_numpy(vector1, vector2)
print(f"Cosine similarity between vector1 and vector2 without numpy: {similarity_score}")

###Cosine similarity between vector1 and vector2 with numpy
def cosine_similarity_with_numpy(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    similarity = dot_product / (norm_a * norm_b)
    return similarity
similarity_score = cosine_similarity_with_numpy(vector1, vector2)
print(f"Cosine similarity between vector1 and vector2 with numpy: {similarity_score}")




