from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset_sites
# Convert text to numerical values
# Train the Naive Bayes model
# Evaluate the model

def pp_data(file):
    data = pd.read_csv(file)
    nltk.download("stopwords")
    nltk.download("punkt")
    stop_words = set(stopwords.words("english"))

    def pp_text(text):
        tokens = word_tokenize(text.lower())
        filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered)

    data["text"] = data["text"].apply(pp_text)
    return data

def create_bag_of_words(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["text"])
    y = data["class"]
    return X, y


def train_model(X, y):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = MultinomialNB()
    classifier.fit(X_trn, y_trn)
    return classifier, X_tst, y_tst

def evaluate_model(classifier, X_tst, y_tst):
    y_pred = classifier.predict(X_tst)
    accuracy = accuracy_score(y_tst, y_pred)
    print("Accuracy of the text predicitng:", accuracy)
    print("Classification review:\n", classification_report(y_tst, y_pred))



data = pp_data("___.csv")
X, y = create_bag_of_words(data)
classifier, X_tst, y_tst = train_model(X, y)
evaluate_model(classifier, X_tst, y_tst)
