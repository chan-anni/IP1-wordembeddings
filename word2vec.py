
import gensim
from gensim.models import Word2Vec

# Load the pretrained Word2Vec model
model_path = "/Users/anni/Library/CloudStorage/OneDrive-EastsidePreparatorySchool/High School/12th Grade/IP/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


# Find similar words
word = 'cat'
similar_words = model.most_similar(word, topn=5)

print(f"Words similar to {word}:")
for word, score in similar_words:
    print(f"{word} - {score}")