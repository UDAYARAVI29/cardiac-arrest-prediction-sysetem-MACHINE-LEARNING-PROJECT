# Import necessary modules
import streamlit as st
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Import necessary functions from web_functions
from web_functions import load_data, predict_decision_tree, predict_kmeans

# Define prediction functions for SVM, Random Forest, Naive Bayes, and Gradient Boosting
def predict_svm(X, y, features):
    """Predict using SVM and return the prediction and accuracy score"""
    model = SVC(probability=True)
    model.fit(X, y)
    prediction = model.predict([features])[0]
    accuracy = model.score(X, y)
    return prediction, accuracy

def predict_random_forest(X, y, features):
    """Predict using Random Forest and return the prediction and accuracy score"""
    model = RandomForestClassifier()
    model.fit(X, y)
    prediction = model.predict([features])[0]
    accuracy = model.score(X, y)
    return prediction, accuracy

def predict_naive_bayes(X, y, features):
    """Predict using Naive Bayes and return the prediction and accuracy score"""
    model = GaussianNB()
    model.fit(X, y)
    prediction = model.predict([features])[0]
    accuracy = model.score(X, y)
    return prediction, accuracy

def predict_gradient_boosting(X, y, features):
    """Predict using Gradient Boosting and return the prediction and accuracy score"""
    model = GradientBoostingClassifier()
    model.fit(X, y)
    prediction = model.predict([features])[0]
    accuracy = model.score(X, y)
    return prediction, accuracy

def app(df, X, y):
    """This function creates the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Decision Tree Classifier</b>, <b style="color:blue">K-Means Clustering</b>, <b style="color:red">SVM</b>, <b style="color:purple">Random Forest</b>, <b style="color:orange">Naive Bayes</b>, and <b style="color:teal">Gradient Boosting</b> for Cardiac Disease Prediction.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user
    age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))
    restbp = st.slider("RestingBP", int(df["RestingBP"].min()), int(df["RestingBP"].max()))
    chol = st.slider("Cholesterol", int(df["Cholesterol"].min()), int(df["Cholesterol"].max()))
    fastbs = st.slider("FastingBS", float(df["FastingBS"].min()), float(df["FastingBS"].max()))
    maxhr = st.slider("MaxHR", float(df["MaxHR"].min()), float(df["MaxHR"].max()))
    oldpeak = st.slider("Oldpeak", int(df["Oldpeak"].min()), int(df["Oldpeak"].max()))

    # Create a list to store all the features
    features = [age, restbp, chol, fastbs, maxhr, oldpeak]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score using Decision Tree
        dt_prediction, dt_accuracy = predict_decision_tree(X, y, features)
        
        # Get prediction using K-Means (accuracy isn't directly applicable here)
        km_prediction = predict_kmeans(X, features)
        
        # Get prediction and model score using SVM
        svm_prediction, svm_accuracy = predict_svm(X, y, features)
        
        # Get prediction and model score using Random Forest
        rf_prediction, rf_accuracy = predict_random_forest(X, y, features)
        
        # Get prediction and model score using Naive Bayes
        nb_prediction, nb_accuracy = predict_naive_bayes(X, y, features)
        
        # Get prediction and model score using Gradient Boosting
        gb_prediction, gb_accuracy = predict_gradient_boosting(X, y, features)
        
        st.info("Prediction Successful...")

        # Print the output according to the prediction
        if dt_prediction == 1:
            st.warning("The person is prone to get cardiac arrest (Decision Tree)!!")
        else:
            st.success("The person is relatively safe from cardiac arrest (Decision Tree)")

        # Note: K-Means prediction output can be customized based on your application's logic
        st.write(f"K-Means Prediction: {km_prediction}")

        # Print the score of the Decision Tree model
        st.write(f"Decision Tree Model Accuracy: {dt_accuracy * 100:.2f}%")

        # Print the score of the SVM model
        st.write(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")

        # Print the score of the Random Forest model
        st.write(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")

        # Print the score of the Naive Bayes model
        st.write(f"Naive Bayes Model Accuracy: {nb_accuracy * 100:.2f}%")

        # Print the score of the Gradient Boosting model
        st.write(f"Gradient Boosting Model Accuracy: {gb_accuracy * 100:.2f}%")
