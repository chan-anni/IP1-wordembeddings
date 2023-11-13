from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine


# model weight files from https://allenai.org/allennlp/software/elmo
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# this is the model
# https://saturncloud.io/glossary/elmo/ code help
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
# The closer to 1 the closer the semantics are, closer to -1 is opposite, 0 is just no semantic similarity
sentence1_vector = embeddings[0].mean(dim=0).numpy()
sentence2_vector = embeddings[1].mean(dim=0).numpy()

similarity = compute_cosine_similarity(sentence1_vector, sentence2_vector)

print(f"Cosine Similarity: {similarity} of {sentences[0]} and {sentences[1]}")



# Sample sentences 2 for visualization
# here I just want to see how semantic similarities can be seen throughout sentences
sentences2 = ["I love programming.", "I hate you.", "I love you.", "The black man was shot", "Martha came to the conclusion that shake weights are a great gift for any occasion.", "She had a difficult time owning up to her own crazy self.", "Help me", "dog", "cat", "She found it hard acknowledging that she was psycho"]
embeddings2 = get_elmo_embeddings(sentences2)


# Averaging word embeddings in each sentence, reducing all of them to one dimension
sentence_embeddings2 = embeddings2.mean(dim=1).detach().numpy()


# Number of samples
n_samples = sentence_embeddings2.shape[0]

# Set perplexity to a value less than n_samples (otherwise throws error)
perplexity_value = min(30, n_samples - 1) 

# reducing down to 2 dimensions with TSNE so that I can plot
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
reduced_embeddings = tsne.fit_transform(sentence_embeddings2)


x = reduced_embeddings[:, 0]
y = reduced_embeddings[:, 1]


# Print out every sentences location as (x,y) coords -> for loop to iterate through sentence array
for i in range(len(x)):  
            print(f"Sentence {i+1}: {sentences2[i]}, x: {x[i]}, y: {y[i]}")
     
# visualizing
plt.figure(figsize=(12, 8))
plt.scatter(x,y)

plt.title("t-SNE visualization of ELMo sentence embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

