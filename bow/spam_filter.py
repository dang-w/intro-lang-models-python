# Naive Bayes classifier

from spam_data import training_spam_docs, training_doc_tokens, training_labels, test_labels, test_spam_docs, training_docs, test_docs
from sklearn.naive_bayes import MultinomialNB

def create_features_dictionary(document_tokens):
  features_dictionary = {}
  index = 0
  for token in document_tokens:
    if token not in features_dictionary:
      features_dictionary[token] = index
      index += 1
  return features_dictionary

def tokens_to_bow_vector(document_tokens, features_dictionary):
  bow_vector = [0] * len(features_dictionary)
  for token in document_tokens:
    if token in features_dictionary:
      feature_index = features_dictionary[token]
      bow_vector[feature_index] += 1
  return bow_vector

bow_sms_dictionary = create_features_dictionary(training_doc_tokens)

training_vectors = [tokens_to_bow_vector(training_doc, bow_sms_dictionary) for training_doc in training_spam_docs]

test_vectors = [tokens_to_bow_vector(test_doc, bow_sms_dictionary) for test_doc in test_spam_docs]

spam_classifier = MultinomialNB()

def spam_or_not(label):
  return "spam" if label else "not spam"

spam_classifier.fit(training_vectors, training_labels)

predictions = spam_classifier.score(test_vectors, test_labels)

# Get a second example index that's within range
second_example_index = min(1, len(test_docs) - 1)  # Use index 1 or the last item if only 1 item exists

print("The predictions for the test data were {0}% accurate.\n\nFor example, '{1}' was classified as {2}.".format(
    predictions * 100,
    test_docs[0],
    spam_or_not(test_labels[0])
))

if len(test_docs) > 1:
    print("\nMeanwhile, '{0}' was classified as {1}.".format(
        test_docs[second_example_index],
        spam_or_not(test_labels[second_example_index])
    ))
