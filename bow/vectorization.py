from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

##########################################################
#from collections import Counter

# tokens = ['another', 'five', 'fish', 'find', 'another', 'faraway', 'fish']
# print(Counter(tokens))
##########################################################

##########################################################
# from sklearn.feature_extraction.text import CountVectorizer

# training_documents = ["Five fantastic fish flew off to find faraway functions.", "Maybe find another five fantastic fish?", "Find my fish with a function please!"]
# test_text = ["Another five fish find another faraway fish."]
# bow_vectorizer = CountVectorizer()
# bow_vectorizer.fit(training_documents)
# bow_vector = bow_vectorizer.transform(test_text)
# print(bow_vector.toarray())
##########################################################

# Counter({'fish': 2, 'another': 2, 'find': 1, 'five': 1, 'faraway': 1})

bow_vectorizer = CountVectorizer()
training_vectors = bow_vectorizer.fit_transform(training_docs)
test_vectors = bow_vectorizer.transform(test_docs)

spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

# Get valid indices that are within range
first_example_index = min(7, len(test_docs) - 1)
second_example_index = min(first_example_index + 1, len(test_docs) - 1)  # Use next index or same if only 1 item

print("The predictions for the test data were {0}% accurate.".format(predictions * 100))
print("\nFor example, '{0}' was classified as {1}.".format(
    test_docs[first_example_index],
    spam_or_not(test_labels[first_example_index])
))

if first_example_index != second_example_index:
    print("\nMeanwhile, '{0}' was classified as {1}.".format(
        test_docs[second_example_index],
        spam_or_not(test_labels[second_example_index])
    ))
