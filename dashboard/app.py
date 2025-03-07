import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import requests
from io import StringIO

@st.cache_data(ttl=600)
def load_data():
    url = "https://github.com/NibroosAbrar/airqualitydashboard/raw/main/dashboard/cleaned_data.csv"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data, parse_dates=[["year", "month", "day", "hour"]])
            df.rename(columns={"year_month_day_hour": "datetime"}, inplace=True)
    
            # Konversi datetime ke tipe yang benar
            df["datetime"] = pd.to_datetime(df["datetime"], format="%Y %m %d %H")
            df["date"] = df["datetime"].dt.date  # Kolom date-only untuk filtering
            st.write("Kolom dalam dataset:", df.columns.tolist())  # Debugging
            return df
        else:
            st.error(f"Gagal mengunduh data, kode status: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        return None

df = load_data()

if df is None:
    st.stop()  # Hentikan aplikasi jika gagal memuat data

st.write("Data berhasil dimuat:", df.head())


st.write("Data berhasil dimuat:", df.head())

# Sidebar filter
st.sidebar.header("Filter Data")

stations = df["station"].unique()
selected_stations = st.sidebar.multiselect("Pilih Stasiun", stations, default=stations)
date_range = st.sidebar.date_input("Rentang Waktu", [df["date"].min(), df["date"].max()])

# Filter data
filtered_df = df[(df["station"].isin(selected_stations)) &
                 (df["date"] >= date_range[0]) &
                 (df["date"] <= date_range[1])]

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
corr_matrix = df[["PM2.5","PM10","SO2","NO2","O3","CO","TEMP","PRES","DEWP","RAIN"]].corr()
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
