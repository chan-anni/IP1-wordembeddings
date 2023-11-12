
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
# I'm looping through the different arrays so that I can speed things up
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

# creating a gender vector for projection
    # blue = male, pink = female
gender_vector = model["he"]-model["she"] 

# here we are testing gender biases in professions
def compute_projection_score(profession, vector_direction, model):
    # getting the embedding for the profession
    profession_vector = model[profession]

     # projecting the profession vector onto the gender direction
    return profession_vector.dot(vector_direction) / np.linalg.norm(vector_direction)

# professions
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction"]
gender_scores = [compute_projection_score(prof, gender_vector, model) for prof in professions]

# visualising!
plt.figure(figsize=(10, 6))
plt.barh(professions, gender_scores, color=['blue' if score > 0 else 'pink' for score in gender_scores])
plt.xlabel("Gender Projection Score")
plt.title("Gender Direction Projection for Different Professions")
plt.grid(axis='x')
plt.xlim(-1.5, 1.5)
plt.show()



# Figure 3: Gender and attributes
attributes = ["nurturing", "loyal", "strong", "kind", "honest", "independent", "leadership", "sexy", "emotional", "calm", "smart", "polite", "confident", "agreeable", "assertive", "passive", "dominant", "competitive", "hardworking", "cute", "talented"]
gender_scores = [compute_projection_score(atr, gender_vector, model) for atr in attributes]


# visualising!
plt.figure(figsize=(15, 10))
plt.barh(attributes, gender_scores, color=['blue' if score > 0 else 'pink' for score in gender_scores])
plt.xlabel("Gender Projection Score")
plt.title("Gender Direction Projection for Different Attributes")
plt.grid(axis='x')
plt.xlim(-1.5, 1.5)
plt.show()




# Race vector because I want to look at race
    # Blue indicates a leaning towards "african"
    # Pink indicates a leaning towards "american"


# Figure 4: Race and professions
race_vector = model["african"]-model["american"] 

# professions
professions = ["doctor", "nurse", "engineer", "teacher", "lawyer","singer", "artist", "librarian", "programmer", "maid", "homemaker", "chef", "CEO", "manager", "lawyer", "paralegal", "attendent", "secretary", "attorny", "athlete", "mechanic", "veteran", "scientist", "salesman", "pitcher", "surgeon", "construction", "criminal", "hairdresser", "quarterback"]
race_scores = [compute_projection_score(prof, race_vector, model) for prof in professions]


plt.figure(figsize=(15, 10))
plt.barh(professions, race_scores, color=['blue' if score > 0 else 'pink' for score in race_scores])
plt.xlabel("Race Projection Score")
plt.title("Race Direction Projection for Different Professions")
plt.grid(axis='x')
plt.xlim(-1.5, 1.5)
plt.show()

# Figure 5: attributes and race
attributes2 = ["violent", "competent", "rational", "sympathetic", "analytical", "novice", "educated", "uneducated", "corrupt", "poor", "trustworthy", "loyal", "poor", "rich", "tall", "forceful", "lazy", "hardworking", "diligent", "intelligent", "gentle", "dangerous"]
race_scores = [compute_projection_score(atr, race_vector, model) for atr in attributes2]


plt.figure(figsize=(15, 10))
plt.barh(attributes2, race_scores, color=['blue' if score > 0 else 'pink' for score in race_scores])
plt.xlabel("Race Projection Score")
plt.title("Figure 6: Race Direction Projection for Different Attributes")
plt.grid(axis='x')
plt.xlim(-1.5, 1.5)
plt.show()



# Figure 6: literally random words and race

random = ["abyss", "flashers", "sweetener", "olives", "chicken", "shins", "cult", "blue", "aprons", "during", "ventilator", "child", "catnip", "brazilians", "pants", "minor", "rice", "rock", "copyright", "likely", "generate", "this"]
race_scores = [compute_projection_score(x, race_vector, model) for x in random]


plt.figure(figsize=(15, 10))
plt.barh(random, race_scores, color=['blue' if score > 0 else 'pink' for score in race_scores])
plt.xlabel("Race Projection Score")
plt.title("Race Direction Projection for Random Words")
plt.grid(axis='x')
plt.xlim(-1.5, 1.5)
plt.show()




