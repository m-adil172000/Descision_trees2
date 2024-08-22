import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.visualizer import plot_decision_tree


#Choosing the dataset and loading it
def user_input_features():
    dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Wine', 'Breast Cancer'))
    return dataset_name

def load_dataset(dataset_name):
    if dataset_name == "Iris":
        data = load_iris()
    elif dataset_name == "Wine":
        data = load_wine()
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
    return data


dataset_name = user_input_features()
data = load_dataset(dataset_name)

X, y = data.data, data.target

st.title("Decision Tree Classifier Hyperparameter Visualizer")
st.sidebar.header("Model Hyperparameters")

#Adding sidebar widgets for hyperparameters
max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 4, 1)
criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
max_features = st.sidebar.selectbox("Max Features", (None, "auto", "sqrt", "log2"))
max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 100, None)
min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", 0.0, 0.5, 0.0)

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training
model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    splitter=splitter,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes,
    min_impurity_decrease=min_impurity_decrease
)
model.fit(X_train, y_train)

# Predictions and Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Dataset: {dataset_name}")
st.write(f"Accuracy: {accuracy:.2f}")

# Plot decision tree
st.subheader("Decision Tree Visualization")
plot_decision_tree(model, data.feature_names, data.target_names)

