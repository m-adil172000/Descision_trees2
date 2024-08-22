import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score


def plot_decision_tree(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, ax=ax)
    st.pyplot(fig)
