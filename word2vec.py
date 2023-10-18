
import gensim
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the pretrained Word2Vec model
model_path = "/Users/anni/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/High School/12th Grade/IP/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)




# Find similar words
# word = 'cat'
# similar_words = model.most_similar(word, topn=5)

result = model.most_similar(positive=['England', 'Paris'], negative=['London'])
print(result)
# print(f"Words similar to {word}:")

# for word, score in similar_words:
#     print(f"{word} - {score}")

# https://stackoverflow.com/questions/40581010/how-to-run-tsne-on-word2vec-created-from-gensim
# for help with visualization

# Take the first 300 words (for simplicity)

# List of specific words to visualize
#specific_words = ["king", "queen", "man", "woman", "computer", "laptop", "coworker", "cat", "dog", "pet", "child", "working", "cats"]

# Extract vectors for the specific words
# word_vectors = np.array([model[word] for word in specific_words if word in model])

# #print(model["hello"])
# # model.wv.most_similar('computer', topn=10)
# # Extract vectors for the specific words and pair them with their labels
# word_label_pairs = [(word, model[word]) for word in specific_words if word in model.key_to_index]

# # Separate the words from vectors so that i can use this for labels later
# words_to_plot = [pair[0] for pair in word_label_pairs]
# vectors_to_plot = np.array([pair[1] for pair in word_label_pairs])

# This is for randomly taking any words                         
#limit = 40
#word_vectors = np.array([model[word] for word in model.index_to_key[:limit]])

# # Apply t-SNE for dimensionality reduction to make it 2D
# tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(word_vectors) - 1))
# Y = tsne.fit_transform(vectors_to_plot)

# # Plot the results
# plt.figure(figsize=(12, 8))
# plt.scatter(Y[:, 0], Y[:, 1])

# for label, x, y in zip(words_to_plot, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

# # plt.show()

