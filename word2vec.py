
import gensim
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the pretrained Word2Vec model
model_path = "/Users/anni/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/High School/12th Grade/IP/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


# Finding similar words
# prints out: noisy, loudly, fer_cryin, deafening, and Loud as results
word = 'loud'
similar_words = model.most_similar(word, topn=10)
print(f"Words similar to {word}:")

for word, score in similar_words:
    print(f"\t{word} - {score}\n")


print("testing vector math:")
# Trying math to see most similar single word
# ex: king + woman - man should be queen
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"\n\t the result is: {result}")




print("\nlooping through arrays to find the most similar")
# seeing that the vector math works, I'm looping through the different arrays so that I can speed things up
# loop to check through traits in relation to man an woman
traits = ["nurturing", "loyal", "strong", "kind", "honest", "independent", "leadership", "sexy", "emotional", "calm", "smart", "polite", "confident", "agreeable", "assertive", "passive", "dominant", "competitive", "hardworking", "cute", "talented"]
for x in traits:
    new_trait = model.most_similar(positive=[x, 'he'], negative=['she'], topn=1)
    print(x + f" - 'she' + 'he': {new_trait}\n")
   

# loop to check through a profession and gender
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction"]
for x in professions:
    new_job = model.most_similar(positive=[x, 'he'], negative=['she'], topn=1)
    print(x + f" - 'she' + 'he': {new_job}\n")
   


# Making graphs and visualizations

# https://stackoverflow.com/questions/40581010/how-to-run-tsne-on-word2vec-created-from-gensim
# for help with visualization

# Figure 1: A 2D graph of word-vectors in relation to each other
#List of specific words to visualize
specific_words = ["king", "queen", "man", "woman", "computer", "laptop", "coworker", "cat", "dog", "pet", "child", "working", "cats", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "leadership", "sexy", "emotional", "calm", "smart", "polite", "confident"]

#Extract vectors for the specific words
word_vectors = np.array([model[word] for word in specific_words if word in model])

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
#plt.xlabel("Gender")
for label, x, y in zip(words_to_plot, Y[:, 0], Y[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show()




# Figure 2: graphing vector projections onto a gendered vector --> seeing if a word leans more male or female

#trying to make a chart for words and seeing their gender more easily

# computing a gender vector
# we want to create a subspace, Gender Subspace, for the larger vector space, V, of all words in Word2Vec
# basically this Gender Subspace has the same properties as V but also specifically defined by the set of vectors that define gender

# however we can only currently calculate one aspect of the gender difference using this:
gender_vector = model["he"]-model["she"] 

# here we are testing gender biases in professions
### replace profession with generic vector name for ease later
def compute_gender_score(profession, gender_direction, model):
    # getting the embedding for the profession
    profession_vector = model[profession]

     # projecting the profession vector onto the gender direction
     # need the dot product of the gender direction vector and the vector making up the word in the profession array
     # normalize/divide by magnitude of the gender direction vector
    return profession_vector.dot(gender_direction) / np.linalg.norm(gender_direction)

# professions
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction"]
gender_scores = [compute_gender_score(prof, gender_vector, model) for prof in professions]

print(gender_scores)

# visualising!
plt.figure(figsize=(10, 6))
plt.barh(professions, gender_scores, color=['blue' if score > 0 else 'pink' for score in gender_scores])
plt.xlabel("Gender Projection Score")
plt.title("Gender Direction Projection for Different Professions")
plt.grid(axis='x')

plt.show()



# attributes
attributes = ["nurturing", "loyal", "strong", "kind", "honest", "independent", "leadership", "sexy", "emotional", "calm", "smart", "polite", "confident", "agreeable", "assertive", "passive", "dominant", "competitive", "hardworking", "cute", "talented"]
gender_scores = [compute_gender_score(atr, gender_vector, model) for atr in attributes]

print(gender_scores)

# visualising!
plt.figure(figsize=(15, 10))
plt.barh(attributes, gender_scores, color=['blue' if score > 0 else 'pink' for score in gender_scores])
plt.xlabel("Gender Projection Score")
plt.title("Gender Direction Projection for Different Attributes")
plt.grid(axis='x')

plt.show()


# Race vector because I want to look at race

race_vector = model["african"]-model["american"] 

professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction"]
gender_scores = [compute_gender_score(job, gender_vector, model) for job in professions]

def compute_race_score(i, race_direction, model):
    # getting the embedding for the profession
    profession_vector = model[i]
    return profession_vector.dot(race_direction) / np.linalg.norm(race_direction)

# professions
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction", "criminal", "hairdresser", "quarterback"]
race_scores = [compute_race_score(prof, race_vector, model) for prof in professions]

print(race_scores)

# Blue indicates a leaning towards "african"
# Pink indicates a leaning towards "american"

plt.figure(figsize=(15, 10))
plt.barh(professions, race_scores, color=['blue' if score > 0 else 'pink' for score in race_scores])
plt.xlabel("Race Projection Score")
plt.title("Race Direction Projection for Different Attributes")
plt.grid(axis='x')

plt.show()


#WEAT Word Embeddings Association Test

A = ["man", "boy", "he", "father", "son", "guy", "male", "uncle"]
B = ["woman", "girl", "she", "mother", "daughter", "female", "aunt"]
X = ["math", "algebra", "science", "programming", "coding", "computer", "data", "calculus", "physics", "engineer", "mechanical", "biology"] # 12 components
Y = ["art", "drawing", "literature", "history", "dance", "poetry", "painting", "reading", "media", "crafts", "writing", "fashion"]

def association_score(word, A, B, X, Y, model):
    return np.mean([model.similarity(word, x) for x in X]) - np.mean([model.similarity(word, y) for y in Y])


def differential_association(A, B, X, Y, model):
    return sum(association_score(word, A, B, X, Y, model) for word in A) - sum(association_score(word, A, B, X, Y, model) for word in B)


def weat_effect_size(A, B, X, Y, model):
    assoc_A = np.mean([association_score(word, A, B, X, Y, model) for word in A])
    assoc_B = np.mean([association_score(word, A, B, X, Y, model) for word in B])
    effect_size = (assoc_A - assoc_B) / np.std([association_score(word, A, B, X, Y, model) for word in A+B])
    return effect_size

def weat_p_value(A, B, X, Y, model, num_permutations=10000):
    observed_statistic = differential_association(A, B, X, Y, model)
    combined = A + B
    count = 0
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_A = combined[:len(A)]
        new_B = combined[len(A):]
        if differential_association(new_A, new_B, X, Y, model) > observed_statistic:
            count += 1
    p_value = count / num_permutations
    return p_value

# Calculate WEAT score
effect_size = weat_effect_size(A, B, X, Y, model)
p_value = weat_p_value(A, B, X, Y, model)

print(f"Effect size: {effect_size}")
print(f"P-value: {p_value}")

# This assumes 'model' is a Word2Vec-like model that has a 'get_vector' method.
words = A + B + X + Y  # Combine all the words you want to visualize
word_vectors = np.array([model.get_vector(word) for word in words])

tsne = TSNE(n_components=2, random_state=0)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Define the size of the plot
plt.figure(figsize=(16, 16))

# Separate the points by their set
for i, word in enumerate(words):
    x, y = word_vectors_2d[i, :]
    plt.scatter(x, y)
    plt.annotate(word, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

# Set words from set A to a specific color
for word in A:
    index = words.index(word)
    plt.annotate(word, xy=(word_vectors_2d[index, 0], word_vectors_2d[index, 1]),
                 bbox=dict(facecolor='green', alpha=0.5))

# Set words from set B to a different color
for word in B:
    index = words.index(word)
    plt.annotate(word, xy=(word_vectors_2d[index, 0], word_vectors_2d[index, 1]),
                 bbox=dict(facecolor='red', alpha=0.5))

# Set words from set X to a different color
for word in X:
    index = words.index(word)
    plt.annotate(word, xy=(word_vectors_2d[index, 0], word_vectors_2d[index, 1]),
                 bbox=dict(facecolor='blue', alpha=0.5))

# Set words from set Y to a different color
for word in Y:
    index = words.index(word)
    plt.annotate(word, xy=(word_vectors_2d[index, 0], word_vectors_2d[index, 1]),
                 bbox=dict(facecolor='yellow', alpha=0.5))

plt.show()
