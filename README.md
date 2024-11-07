# Independent Project 1 - Semantic Representations and Bias Through Word Embeddings

This independent project explores how word embeddings capture semantic relationships between words in natural language processing (NLP) and investigates how these embeddings also reflect underlying biases, particularly regarding gender and race. Through first understanding how word embeddings work then by examining popular word embedding models, specifically Word2Vec and ELMo, this project aims to reveal how these models learn associations between words with similar meanings and carry human biases present in the training data. 

This project is motivated by a curiosity about how machines represent human language and an interest in the societal implications of these biases.

### Goals
Understand Word Relationships: Explore how embeddings represent words in different contexts and capture semantic relationships.
Analyze Bias: Investigate how embeddings encode gender and racial biases, and examine methods for quantifying and visualizing these biases.

### Data sources
- Pre-trained Word2Vec embeddings from Google News dataset
- ELMo model from allennlp
- Model weight files for ELMo from: https://allenai.org/allennlp/software/elmo


### Results
**Gender Bias**: Observed some strong projections of stereotypical gender roles within certain word groups (e.g., “nurse” closer to female-associated terms).
![image](https://github.com/user-attachments/assets/4be37204-743c-4e22-942c-d0d2b3e8c31b)
<img width="949" alt="image" src="https://github.com/user-attachments/assets/0bfc6bf1-e996-4750-87ae-974546813454">

**Racial Bias**: Observed some projections of cultural contextual biases relating to race based on names and other racially related words (e.g., comparing the most common black and white name to a set of words, "athlete" leans more towards the name Deshawn over Jake).
<img width="977" alt="image" src="https://github.com/user-attachments/assets/e586822d-acec-433e-a548-02d1ee6adf9d">


Setup: 
-----------------------------------------------------
download and install: 
pip install -r requirements.txt


 Setup Python 3.9.18 environment for the ELMo model 


 Setup Python 3.11 environment for "word2vec" 
