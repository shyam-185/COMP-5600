import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

# Output of program
def set_display_options(max_colwidth=1000, max_rows=10):
    pd.set_option('display.max_colwidth', max_colwidth)
    pd.set_option('display.max_rows', max_rows)

# Loads the dataset
dataset = load_dataset('rotten_tomatoes')

# Sentiment score getter
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Preprocesses and labels encode target
data = pd.DataFrame(dataset['train']['text'], columns=['review'])
data['sentiment_score'] = data['review'].apply(get_sentiment)

# If polarity > 0.1 then 'positive', < -0.1 then 'negative', between -0.1 and 0.1 then 'conflicted'
data['label'] = pd.cut(data['sentiment_score'], bins=[-np.inf, -0.1, 0.1, np.inf], labels=['negative', 'conflicted', 'positive'])

# Features and Labels
X = data['review']
y = data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Splits the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Vectorization with n-grams and word count
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), analyzer='word')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_vectors, y_train)

# Multi-layer Perceptron classifier
mlp_model = MLPClassifier(random_state=1, max_iter=300)
mlp_model.fit(X_train_vectors, y_train)

# This function searches the reviews based on the user's query and then displays the results
def search_reviews(query, model, vectorizer, le):
    # Adjusts the display settings
    set_display_options(max_colwidth=100, max_rows=10)
    
    # Filters the reviews containing the query
    mask = data['review'].str.lower().str.contains(query.lower())
    filtered_reviews = data[mask]

    # Check if any reviews were found
    if filtered_reviews.empty:
        return "No reviews found for the given query."

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

# Keeps prompting until quit
while True:
    # Ask for user input
    query = input("\nWhat are you looking for? Enter an actor, movie, or other related keywords (or type 'quit' to quit): ")
    
    # Check if the user wants to quit
    if query.lower() == 'quit':
        print("Exiting the program.")
        break

    # Search and print results using Logistic Regression model
    logistic_results = search_reviews(query, logistic_model, vectorizer, le)
    print("\nSearch Results using Logistic Regression Model:")
    print(logistic_results)

    # Search and print results using MLP model
    mlp_results = search_reviews(query, mlp_model, vectorizer, le)
    print("\nSearch Results using Multi-layer Perceptron Model:")
    print(mlp_results)

# Baseline Performance
dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X_train_vectors, y_train)
y_random_pred = dummy_clf.predict(X_test_vectors)
print("\nRandom Baseline Performance:")
print(classification_report(y_test, y_random_pred, target_names=le.classes_))

# Logistic Regression Model Performance
y_logistic_pred = logistic_model.predict(X_test_vectors)
print("\nLogistic Regression Model's Performance:")
print(classification_report(y_test, y_logistic_pred, target_names=le.classes_))

# MLP Model Performance
y_mlp_pred = mlp_model.predict(X_test_vectors)
print("\nMulti-layer Perceptron Model's Performance:")
print(classification_report(y_test, y_mlp_pred, target_names=le.classes_))
