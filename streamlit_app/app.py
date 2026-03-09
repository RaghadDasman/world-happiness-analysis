import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="World Happiness Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 World Happiness Dashboard")
st.write("Welcome to my interactive dashboard for exploring global happiness data from 2015 to 2019 😊")

df = pd.read_csv("../data/clean_happiness_data.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📏 Dataset Shape")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")