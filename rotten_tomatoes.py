import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier

# Load the dataset
dataset = load_dataset('rotten_tomatoes')

# Function to calculate sentiment scores
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Preprocess and label encode target
data = pd.DataFrame(dataset['train']['text'], columns=['review'])
data['sentiment_score'] = data['review'].apply(get_sentiment)

# Assuming polarity > 0.1 as positive, < -0.1 as negative, between -0.1 and 0.1 as conflicted
data['label'] = pd.cut(data['sentiment_score'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['negative', 'conflicted', 'positive'])

# Features and Labels
X = data['review']
y = data['label']

# Encoding labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorization with n-grams and word count
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), analyzer='word')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)

# Function to search reviews based on user query and display results
def search_reviews(query):
    # Filter reviews containing the query
    mask = data['review'].str.lower().str.contains(query.lower())
    filtered_reviews = data[mask]
    # Predict sentiment
    filtered_vectors = vectorizer.transform(filtered_reviews['review'])
    predictions = model.predict(filtered_vectors)
    classes = le.inverse_transform(predictions)
    # Display results
    results = pd.DataFrame({
        'Review': filtered_reviews['review'],
        'Class': classes,
        'Sentiment Score': filtered_reviews['sentiment_score']
    })
    return results

# User interaction
query = input("What are you looking for? Enter an actor, movie, or other related keywords: ")
results = search_reviews(query)
print(results)

# baseline :

# Assuming 'y_train' is your training set labels
class_counts = np.bincount(y_train)
class_probabilities = class_counts / class_counts.sum()

# Create a dummy classifier that predicts randomly based on class distributions
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train_vectors, y_train)

# Predict on the test set
y_random_pred = dummy_clf.predict(X_test_vectors)

# Evaluation
print("Random Baseline Performance:")
print(classification_report(y_test, y_random_pred, target_names=['negative', 'conflicted', 'positive']))


# preliminary
# Predict on the test set using the trained model
y_pred = model.predict(X_test_vectors)

# Evaluation of your actual model
print("Your Model's Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

