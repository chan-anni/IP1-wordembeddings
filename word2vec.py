
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

result = model.most_similar(positive=['softball', 'he'], negative=['she'])
print(result)

print("\n")

result2 = model.most_similar(positive=['baseball', 'she'], negative=['he'])

print(result2)


# print(f"Words similar to {word}:")

# for word, score in similar_words:
#     print(f"{word} - {score}")

# https://stackoverflow.com/questions/40581010/how-to-run-tsne-on-word2vec-created-from-gensim
# for help with visualization

# Take the first 300 words (for simplicity)

#List of specific words to visualize
specific_words = ["king", "queen", "man", "woman", "computer", "laptop", "coworker", "cat", "dog", "pet", "child", "working", "cats"]

#Extract vectors for the specific words
word_vectors = np.array([model[word] for word in specific_words if word in model])

# #print(model["hello"])
# # model.wv.most_similar('computer', topn=10)
# # Extract vectors for the specific words and pair them with their labels
word_label_pairs = [(word, model[word]) for word in specific_words if word in model.key_to_index]

# Separate the words from vectors so that i can use this for labels later
words_to_plot = [pair[0] for pair in word_label_pairs]
vectors_to_plot = np.array([pair[1] for pair in word_label_pairs])

#This is for randomly taking any words                         
limit = 40
word_vectors = np.array([model[word] for word in model.index_to_key[:limit]])



## Plotting out specific words out to reference and see how it works

# Apply t-SNE for dimensionality reduction to make it 2D
tsne = TSNE(n_components=2, random_state=0, perplexity=min(5, len(word_vectors) - 1))
Y = tsne.fit_transform(vectors_to_plot)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(Y[:, 0], Y[:, 1])

for label, x, y in zip(words_to_plot, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

#plt.show()


#trying to make a chart for words and seeing their gender more easily

# computing a gender vector
# we want to create a subspace, Gender Subspace, for the larger vector space, V, of all words in Word2Vec
# basically this Gender Subspace has the same properties as V but also specifically defined by the set of vectors that define gender

# however we can only currently calculate one aspect of the gender difference using this:
gender_vector = model["he"]-model["she"] # try PCA later if possible

# here we are testing gender biases in professions
def compute_gender_score(profession, gender_direction, model):
    # getting the embedding for the profession
    profession_vector = model[profession]

     # projecting the profession vector onto the gender direction
     # need the dot product of the gender direction vector and the vector making up the word in the profession array
     # normalize/divide by magnitude of the gender direction vector
    return profession_vector.dot(gender_direction) / np.linalg.norm(gender_direction)

# professions
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran"]
gender_scores = [compute_gender_score(prof, gender_vector, model) for prof in professions]

print(gender_scores)

# visualising!
plt.figure(figsize=(10, 6))
plt.barh(professions, gender_scores, color=['blue' if score > 0 else 'pink' for score in gender_scores])
plt.xlabel("Gender Projection Score")
plt.title("Gender Direction Projection for Different Professions")
plt.grid(axis='x')

plt.show()