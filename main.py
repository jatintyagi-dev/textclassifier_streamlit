import pandas as pd
import streamlit as st
import fastai
from train import train_lm


st.title("Text Classifier!")

file = st.file_uploader("Please upload the data for training ")


if file:
    inp = pd.DataFrame(file)
    st.dataframe(inp)


train = st.sidebar.selectbox("Pick a process", ["Training ", "Inference"])

if train == "Training":

    LM = st.sidebar.selectbox("Train a Language Model", ["Yes ", "No"])

    if LM == "Yes":
        lmlearner = train_lm()
