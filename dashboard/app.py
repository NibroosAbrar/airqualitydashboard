import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style='dark')

# Load cleaned data
merged_df = pd.read_csv("cleaned_data.csv")

# Convert to datetime format
merged_df["datetime"] = pd.to_datetime(merged_df[["year", "month", "day", "hour"]])

# Trend analysis: Average PM2.5 over time for each station
st.title("Analisis Kualitas Udara")
st.subheader("Tren Kualitas Udara dari Waktu ke Waktu")

daily_trend = merged_df.groupby(["datetime", "station"])["PM2.5"].mean().reset_index()
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(data=daily_trend, x="datetime", y="PM2.5", hue="station", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Correlation analysis to find main influencing factors
st.subheader("Faktor yang Mempengaruhi Kualitas Udara")
corr_matrix = merged_df[["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Air quality comparison across stations
st.subheader("Perbandingan Kualitas Udara Antar Stasiun")
fig, ax = plt.subplots(figsize=(15, 6))
sns.boxplot(data=merged_df, x="station", y="PM2.5")
plt.xticks(rotation=45)
st.pyplot(fig)
