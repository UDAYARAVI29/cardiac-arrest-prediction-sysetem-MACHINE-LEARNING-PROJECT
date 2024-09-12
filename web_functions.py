"""This module contains necessary functions"""

# Import necessary modules
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import streamlit as st


@st.cache()
def load_data():
    """This function returns the preprocessed data"""
    # Load the Heart Disease dataset into DataFrame
    df = pd.read_csv('heart.csv')

    # Perform feature and target split
    X = df[["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]]
    y = df['HeartDisease']

    return df, X, y


@st.cache()
def train_decision_tree(X, y):
    """This function trains a Decision Tree model and returns the model and its accuracy"""
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return model, accuracy


@st.cache()
def train_random_forest(X, y):
    """This function trains a Random Forest model and returns the model and its accuracy"""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return model, accuracy


@st.cache()
def train_svm(X, y):
    """This function trains an SVM model and returns the model and its accuracy"""
    model = SVC(probability=True, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return model, accuracy


@st.cache()
def train_naive_bayes(X, y):
    """This function trains a Naive Bayes model and returns the model and its accuracy"""
    model = GaussianNB()
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return model, accuracy


@st.cache()
def train_kmeans(X):
    """This function trains a K-Means clustering model and returns the model"""
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)
    return model


def predict_decision_tree(X, y, features):
    """This function makes predictions using the Decision Tree model"""
    model, accuracy = train_decision_tree(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, accuracy


def predict_random_forest(X, y, features):
    """This function makes predictions using the Random Forest model"""
    model, accuracy = train_random_forest(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, accuracy


def predict_svm(X, y, features):
    """This function makes predictions using the SVM model"""
    model, accuracy = train_svm(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, accuracy


def predict_naive_bayes(X, y, features):
    """This function makes predictions using the Naive Bayes model"""
    model, accuracy = train_naive_bayes(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, accuracy


def predict_kmeans(X, features):
    """This function makes predictions using the K-Means model"""
    model = train_kmeans(X)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction


def build_lstm(num_units, input_shape):
    """This function builds an LSTM model"""
    input_layer = tf.keras.layers.Input(shape=input_shape)
    lstm_layer = tf.keras.layers.LSTM(num_units, return_sequences=True)(input_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_layer)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
    return model


# Example usage:
if __name__ == "__main__":
    # Load data
    df, X, y = load_data()

    # Example features for prediction
    features = [55, 140, 250, 0, 150, 2.5]

    # Predict using Decision Tree
    dt_prediction, dt_accuracy = predict_decision_tree(X, y, features)
    st.write(f"Decision Tree Prediction: {dt_prediction}")
    st.write(f"Decision Tree Accuracy: {dt_accuracy}")

    # Predict using Random Forest
    rf_prediction, rf_accuracy = predict_random_forest(X, y, features)
    st.write(f"Random Forest Prediction: {rf_prediction}")
    st.write(f"Random Forest Accuracy: {rf_accuracy}")

    # Predict using SVM
    svm_prediction, svm_accuracy = predict_svm(X, y, features)
    st.write(f"SVM Prediction: {svm_prediction}")
    st.write(f"SVM Accuracy: {svm_accuracy}")

    # Predict using Naive Bayes
    nb_prediction, nb_accuracy = predict_naive_bayes(X, y, features)
    st.write(f"Naive Bayes Prediction: {nb_prediction}")
    st.write(f"Naive Bayes Accuracy: {nb_accuracy}")

    # Predict using K-Means
    km_prediction = predict_kmeans(X, features)
    st.write(f"K-Means Prediction: {km_prediction}")
