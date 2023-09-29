from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
import numpy as np
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
sentences = ["Cats", "London."]
embeddings = get_elmo_embeddings(sentences)

# Taking the mean of embeddings of all words in a sentence to get a sentence vector
sentence1_vector = embeddings[0].mean(dim=0).numpy()
sentence2_vector = embeddings[1].mean(dim=0).numpy()

similarity = compute_cosine_similarity(sentence1_vector, sentence2_vector)

print(f"Cosine Similarity: {similarity}")
