import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Load data dengan caching
@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/NibroosAbrar/airqualitydashboard/main/dashboard/cleaned_data.csv"
    
    # Baca CSV tanpa parse_dates
    df = pd.read_csv(url)
    
    # Gabungkan kolom menjadi datetime
    df["datetime"] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')

    # Buat kolom date-only untuk filtering
    df["date"] = df["datetime"].dt.date

    return df

df = load_data()

# Sidebar filter
st.sidebar.header("Filter Data")

stations = df["station"].unique()
selected_stations = st.sidebar.multiselect("Pilih Stasiun", stations, default=stations)

# Atur rentang waktu agar tidak crash jika ada error
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Rentang Waktu", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter data dengan pengecekan agar tidak error
filtered_df = df[(df["station"].isin(selected_stations)) & 
                 (df["date"] >= date_range[0]) & 
                 (df["date"] <= date_range[1])]

if not filtered_df.empty:  # Pastikan ada data sebelum memplot
    # 1️⃣ Tren Kualitas Udara
    st.subheader("📊 Tren Kualitas Udara per Stasiun")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(data=filtered_df, x="datetime", y="PM2.5", hue="station", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Tanggal")
    plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
    plt.title("Tren PM2.5 dari Waktu ke Waktu")
    st.pyplot(fig)

    # 2️⃣ Faktor yang Mempengaruhi Kualitas Udara
    st.subheader("🔍 Faktor yang Mempengaruhi Kualitas Udara")
    corr_matrix = df[["PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "TEMP", "PRES", "DEWP", "RAIN"]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Korelasi Antar Parameter Kualitas Udara")
    st.pyplot(fig)

    # 3️⃣ Perbandingan Kualitas Udara Antarstasiun
    st.subheader("📌 Perbandingan Kualitas Udara Antarstasiun")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=filtered_df, x="station", y="PM2.5")
    plt.xticks(rotation=45)
    plt.xlabel("Stasiun")
    plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
    plt.title("Distribusi PM2.5 di Berbagai Stasiun")
    st.pyplot(fig)

    # 4️⃣ Forecasting dengan ARIMA
    st.subheader("📈 Forecasting PM2.5 dengan ARIMA")
    selected_station = st.selectbox("Pilih Stasiun untuk Forecasting", stations)
    station_df = df[df["station"] == selected_station].set_index("datetime")["PM2.5"].resample("D").mean().fillna(method="ffill")

    if station_df.dropna().shape[0] > 30:  # Pastikan ada cukup data untuk ARIMA
        model = ARIMA(station_df.dropna(), order=(5, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

        # Plot forecasting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(station_df, label="Actual Data")
        ax.plot(forecast, label="Forecast (30 Hari ke Depan)", linestyle="dashed", color="red")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"Forecasting PM2.5 untuk Stasiun {selected_station}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Data tidak cukup untuk melakukan forecasting dengan ARIMA.")
else:
    st.warning("Tidak ada data yang tersedia dalam rentang waktu dan stasiun yang dipilih.")
