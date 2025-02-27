import re
import random

# Sample spam and non-spam messages
spam_messages = [
    "URGENT: You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
    "Congratulations! You've won a £1000 cash prize! Call now to claim",
    "PRIVATE! Your 2004 Account Statement for shows 800 un-redeemed S.I.M. points. Call 08719899217 Identifier Code:",
    "URGENT! We are trying to contact you. Last weekends draw shows that you have won a £900 prize GUARANTEED",
    "URGENT! Your Mobile No was awarded a £2,000 Bonus Caller Prize on 5/9/03! This is our 2nd attempt to contact YOU!",
    "FREE RINGTONE text FIRST to 87131 for a poly or text GET to 87131 for a true tone!",
    "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw",
    "Dear 1. Your mobile No. was recently awarded a £2000 Bonus Prize. 2 claim is easy, just call 087104711148",
    "URGENT! Your Mobile number has been awarded with a £2000 prize GUARANTEED",
    "URGENT! Your Mobile No was awarded a £2,000 Bonus Caller Prize on 1/08/03! This is our 2nd attempt to contact YOU!",
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
    "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
    "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575."
]

non_spam_messages = [
    "I'll be there in 10 minutes",
    "Sorry, I can't talk right now. I'll call you later.",
    "What time is the meeting tomorrow?",
    "Don't forget to pick up milk on your way home",
    "Hey, how's it going?",
    "The report is ready for your review",
    "I'm running late for dinner, start without me",
    "Can you send me the document we discussed?",
    "Happy birthday! Hope you have a great day",
    "The weather looks nice for the weekend",
    "I'm at the store, do you need anything?",
    "Just checking in to see how you're doing",
    "The movie starts at 8pm, let's meet at 7:30",
    "Don't forget we have dinner with the Johnsons on Friday",
    "I'll be there in 10 minutes"
]

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Split into tokens
    tokens = text.split()
    return tokens

# Create datasets
all_docs = spam_messages + non_spam_messages
all_labels = [1] * len(spam_messages) + [0] * len(non_spam_messages)

# Shuffle data while keeping labels aligned
combined = list(zip(all_docs, all_labels))
random.seed(42)  # For reproducibility
random.shuffle(combined)
all_docs, all_labels = zip(*combined)

# Split into training and test sets (70% training, 30% test)
split_point = int(len(all_docs) * 0.7)
training_docs = all_docs[:split_point]
training_labels = all_labels[:split_point]
test_docs = all_docs[split_point:]
test_labels = all_labels[split_point:]

# Preprocess documents
training_spam_docs = [preprocess_text(doc) for doc in training_docs]
test_spam_docs = [preprocess_text(doc) for doc in test_docs]

# Create a set of all unique tokens in the training data
training_doc_tokens = set()
for doc in training_spam_docs:
    training_doc_tokens.update(doc)
training_doc_tokens = list(training_doc_tokens)