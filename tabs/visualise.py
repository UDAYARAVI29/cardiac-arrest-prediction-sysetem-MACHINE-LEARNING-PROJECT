"""This module contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import streamlit as st

# Import necessary functions from web_functions
from web_functions import train_decision_tree

def app(df, X, y):
    """This function creates the visualization page"""
    
    # Remove warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise Heart Ailment Demographics")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        # Select numeric columns for correlation heatmap
        numeric_cols = X.select_dtypes(include='number').columns
        df_numeric = df[numeric_cols]

        fig_corr = plt.figure(figsize=(8, 6))
        ax_corr = sns.heatmap(df_numeric.corr(), annot=True)
        bottom_corr, top_corr = ax_corr.get_ylim()
        ax_corr.set_ylim(bottom_corr + 0.5, top_corr - 0.5)
        st.pyplot(fig_corr)
        plt.close(fig_corr)  # Close the figure to avoid overlap

    # Scatter plot of RestingBP vs Age
    if st.checkbox("RestingBP vs Age Plot"):
        st.subheader("RestingBP vs Age Plot")

        fig_scatter = plt.figure(figsize=(8, 6))
        sns.scatterplot(x="Age", y="RestingBP", data=df)
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)  # Close the figure to avoid overlap

    # Pie chart of sample results
    if st.checkbox("Show Sample Results"):
        st.subheader("Sample Results")

        safe = (df['HeartDisease'] == 0).sum()
        prone = (df['HeartDisease'] == 1).sum()
        data = [safe, prone]
        labels = ['Safe', 'Prone']
        colors = sns.color_palette('pastel')[0:7]
        fig_pie = plt.figure(figsize=(6, 6))
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        st.pyplot(fig_pie)
        plt.close(fig_pie)  # Close the figure to avoid overlap

    # Plot Decision Tree
    if st.checkbox("Plot Decision Tree"):
        st.subheader("Decision Tree Plot")

        model, accuracy = train_decision_tree(X, y)
        # Export decision tree in dot format and store in 'dot_data' variable
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=X.columns, class_names=['0', '1']
        )
        # Plot the decision tree using graphviz_chart function of streamlit
        st.graphviz_chart(dot_data)

