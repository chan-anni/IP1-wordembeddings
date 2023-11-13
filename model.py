from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine


# Make sure you've downloaded the weights and options files
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)

def get_elmo_embeddings(sentences):
    character_ids = batch_to_ids(sentences)
    with torch.no_grad():
        embeddings = elmo(character_ids)
    return embeddings['elmo_representations'][0]

def compute_cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)


# Example sentences to test for cosine similarity
sentences = ["The police officer ate", "The police man ate"]
embeddings = get_elmo_embeddings(sentences)


# Taking the mean of embeddings of all words in a sentence to get a sentence vector
# The closer to one the closer the semantics are
sentence1_vector = embeddings[0].mean(dim=0).numpy()
sentence2_vector = embeddings[1].mean(dim=0).numpy()

similarity = compute_cosine_similarity(sentence1_vector, sentence2_vector)

print(f"Cosine Similarity: {similarity}")



# Sample sentences
sentences = ["I love programming.", "Coding is fun.", "Python is my favorite language."]



# Averaging word embeddings in each sentence
sentence_embeddings = embeddings.mean(dim=1).detach().numpy()

print("Number of samples:", sentence_embeddings.shape[0])

# Number of samples
n_samples = sentence_embeddings.shape[0]

# Set perplexity to a value less than n_samples
perplexity_value = min(30, n_samples - 1)  # Adjust as needed

tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
reduced_embeddings = tsne.fit_transform(sentence_embeddings)


x = reduced_embeddings[:, 0]
y = reduced_embeddings[:, 1]

# Scatter plot
plt.scatter(x, y)

# Adding labels
for i, txt in enumerate(sentences):  # 'sentences' is your list of original sentences
    plt.annotate(txt, (x[i], y[i]))
plt.figure(figsize=(12, 8))

plt.title("t-SNE visualization of ELMo sentence embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
