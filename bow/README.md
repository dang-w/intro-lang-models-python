# Introduction to Language Models with Python

This repository contains code examples for an introductory course on language models using Python. The examples focus on the Bag of Words (BOW) approach and its application to text classification, specifically spam detection.

## Key Learnings About Bag of Words

- Bag-of-words (BoW) — also referred to as the unigram model — is a statistical language model based on word count.
- There are loads of real-world applications for BoW.
- BoW can be implemented as a Python dictionary with each key set to a word and each value set to the number of times that word appears in a text.
- For BoW, training data is the text that is used to build a BoW model.
- BoW test data is the new text that is converted to a BoW vector using a trained features dictionary.
- A feature vector is a numeric depiction of an item's salient features.
- Feature extraction (or vectorization) is the process of turning text into a BoW vector.
- A features dictionary is a mapping of each unique word in the training data to a unique index. This is used to build out BoW vectors.
- BoW has less data sparsity than other statistical models. It also suffers less from overfitting.
- BoW has higher perplexity than other models, making it less ideal for language prediction.
- One solution to overfitting is language smoothing, in which a bit of probability is taken from known words and allotted to unknown words.

*Note: The spam data for this lesson were taken from the UCI Machine Learning Repository.*

## What is a Language Model?

A language model is a computational model that can understand, interpret, and generate human language. These models are the foundation of many natural language processing (NLP) applications like:

- Text classification (spam detection, sentiment analysis)
- Machine translation
- Text generation
- Question answering
- And many more

## Bag of Words (BOW) Approach

The Bag of Words model is one of the simplest approaches to represent text data for machine learning algorithms. Despite its simplicity, it's quite effective for many text classification tasks.

### Key Concepts

1. **Tokenization**: Breaking text into individual words or tokens
2. **Dictionary Creation**: Creating a vocabulary of unique words from the corpus
3. **Vector Representation**: Converting text into numerical vectors based on word frequencies
4. **Feature Engineering**: Using these vectors as features for machine learning models

## Files in this Repository

#### Basic Text Preprocessing
- `preprocessing.py`: Contains a simple function to preprocess text by converting to lowercase and splitting into tokens

#### Bag of Words Implementation
- `base_bow.py`: Basic implementation of the Bag of Words model using a dictionary to count word frequencies
- `bow_vector.py`: Creates a features dictionary from a corpus of documents
- `bow_vector_cont.py`: Converts text to a BOW vector using the features dictionary

#### Spam Filter Application
- `spam_data.py`: Contains sample spam and non-spam messages, with preprocessing functions
- `spam_filter.py`: Implements a Naive Bayes classifier for spam detection using the BOW approach
- `vectorization.py`: Demonstrates how to use scikit-learn's CountVectorizer for text vectorization

#### Advanced Language Model Concepts
- `bigrams.py`: Implements a simple bigram language model
- `perplexity.py`: Demonstrates how to calculate perplexity for language models
- `document.py`: Contains sample text data (Oscar Wilde quotes) for language model examples

#### Documentation
- `bow-lm.pdf`: Detailed documentation on Bag of Words language models

## How the Spam Filter Works

1. **Data Preparation**:
   - Sample messages are labeled as spam (1) or not spam (0)
   - Data is shuffled and split into training and test sets
   - Text is preprocessed (lowercase, remove punctuation, tokenize)

2. **Feature Extraction**:
   - A dictionary of all unique words in the training data is created
   - Each message is converted to a vector where each position represents a word
   - The value at each position is the count of that word in the message

3. **Classification**:
   - A Multinomial Naive Bayes classifier is trained on the vectors
   - The model learns which words are more likely to appear in spam vs. non-spam messages
   - New messages are classified based on their word frequencies

## Key Machine Learning Concepts Used

### Naive Bayes Classification

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It's called "naive" because it assumes that features (words in our case) are independent of each other, which is a simplification but works well for text classification.

The classifier calculates:
- The probability of a message being spam given its words
- The probability of each word appearing in spam vs. non-spam messages

### Vectorization

Converting text to numerical vectors is essential for machine learning algorithms. In our implementation:
- Each position in the vector corresponds to a unique word
- The value represents how many times that word appears in the message

### Model Evaluation

The spam filter evaluates its performance by:
- Splitting data into training and test sets
- Training on the training set
- Measuring accuracy on the test set

## Implementation Details

### Text Preprocessing

```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Split into tokens (words)
    tokens = text.split()
    return tokens
```

More advanced preprocessing might include:
- Removing punctuation
- Removing numbers
- Removing stop words
- Stemming/lemmatization

### Basic BOW Implementation

```python
def text_to_bow(some_text):
    bow_dictionary = {}
    tokens = preprocess_text(some_text)

    for token in tokens:
        if token in bow_dictionary:
            bow_dictionary[token] += 1
        else:
            bow_dictionary[token] = 1

    return bow_dictionary
```

### Creating a Features Dictionary

```python
def create_features_dictionary(documents):
    features_dictionary = {}

    merged = ' '.join(documents)
    tokens = preprocess_text(merged)

    index = 0
    for token in tokens:
        if token not in features_dictionary:
            features_dictionary[token] = index
            index += 1

    return features_dictionary
```

### Converting Text to BOW Vectors

```python
def text_to_bow_vector(some_text, features_dictionary):
    bow_vector = [0] * len(features_dictionary)
    tokens = preprocess_text(some_text)

    for token in tokens:
        if token in features_dictionary:
            feature_index = features_dictionary[token]
            bow_vector[feature_index] += 1

    return bow_vector
```

### Using Python Libraries

#### Counter from collections

```python
from collections import Counter

tokens = ['fish', 'fish', 'bird', 'cat', 'bird']
print(Counter(tokens))  # Output: Counter({'fish': 2, 'bird': 2, 'cat': 1})
```

#### CountVectorizer from scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

# Create and fit the vectorizer
bow_vectorizer = CountVectorizer()
bow_vectorizer.fit(training_documents)

# Transform documents to vectors
training_vectors = bow_vectorizer.transform(training_documents)
test_vectors = bow_vectorizer.transform(test_documents)
```

#### Text Classification with Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB

# Create and train the classifier
spam_classifier = MultinomialNB()
spam_classifier.fit(training_vectors, training_labels)

# Make predictions
predictions = spam_classifier.predict(test_vectors)

# Evaluate accuracy
accuracy = spam_classifier.score(test_vectors, test_labels)
print(f"Accuracy: {accuracy * 100}%")
```

## Limitations of BOW

- Ignores word order and context
- Creates very high-dimensional, sparse vectors
- Doesn't capture semantics or meaning
- Struggles with out-of-vocabulary words

## Advanced Techniques

- **TF-IDF**: Term Frequency-Inverse Document Frequency weighting
- **N-grams**: Including sequences of N words instead of single words
- **Word Embeddings**: Dense vector representations that capture semantic meaning (Word2Vec, GloVe)
- **Neural Language Models**: More sophisticated models like RNNs, LSTMs, and Transformers

## Getting Started

1. Make sure you have Python 3.6+ installed
2. Install required packages: `pip install scikit-learn`
3. Run the examples in sequence to understand the progression:
   ```
   python preprocessing.py
   python base_bow.py
   python bow_vector.py
   python bow_vector_cont.py
   python spam_filter.py
   ```

## Further Learning

After understanding these basics, you might want to explore:
- More sophisticated text preprocessing (stemming, lemmatization)
- TF-IDF (Term Frequency-Inverse Document Frequency) weighting
- Word embeddings (Word2Vec, GloVe)
- Neural network-based language models (RNNs, Transformers)
- Pre-trained models like BERT, GPT, etc.

## Resources

- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Stanford NLP Course](https://web.stanford.edu/class/cs224n/)
