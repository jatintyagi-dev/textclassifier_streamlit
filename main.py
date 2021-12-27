import pandas as pd
import streamlit as st
import fastai
from train import train_lm, train_cls


st.title("Text Classifier!")

file = st.file_uploader("Please upload the data for training ")


if file:
    inp = pd.read_csv(file)
    st.dataframe(inp)

    inp = inp[["Text", "Sentiment"]]


train = st.sidebar.selectbox(
    "Pick a process", ["Training LM", "Training Classifier"])

if train == "Training LM":

    LM = st.sidebar.selectbox("Train a Language Model", ["Yes ", "No"])

    if LM == "Yes":
        lmlearner = train_lm(inp)
        model_trained = train_cls(inp, train_lm=True)
        st.header("Model Training Finished")

else:
    model_trained = train_cls(inp, train_lm=False)
    st.header("Model Training Finished")
