import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/NibroosAbrar/airqualitydashboard/main/dashboard/cleaned_data.csv"
    df = pd.read_csv(url)
    
    # Debugging: Tampilkan kolom yang ada
    st.write("Kolom dalam dataset:", df.columns.tolist())

    # Cek apakah 'year', 'month', 'day', 'hour' tersedia
    expected_columns = {'year', 'month', 'day', 'hour'}
    if expected_columns.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
    elif "datetime" in df.columns:  # Jika 'datetime' sudah ada
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        st.error("Kolom datetime tidak ditemukan dalam dataset.")
        return None  # Return None jika gagal

    df["date"] = df["datetime"].dt.date
    return df

# Coba memuat data
df = load_data()

if df is None:
    st.error("Gagal memuat data. Periksa kembali struktur dataset.")
    st.stop()  # Hentikan eksekusi jika df None

# Sidebar filter
st.sidebar.header("Filter Data")

stations = df["station"].unique() if "station" in df.columns else []
selected_stations = st.sidebar.multiselect("Pilih Stasiun", stations, default=stations)

# Atur rentang waktu agar tidak error
if "date" in df.columns:
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Rentang Waktu", [min_date, max_date], min_value=min_date, max_value=max_date)
else:
    st.error("Kolom 'date' tidak ditemukan.")
    st.stop()

# Filter data dengan pengecekan agar tidak error
filtered_df = df[(df["station"].isin(selected_stations)) & 
                 (df["date"] >= date_range[0]) & 
                 (df["date"] <= date_range[1])]

if not filtered_df.empty:
    st.subheader("📊 Tren Kualitas Udara per Stasiun")
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.lineplot(data=filtered_df, x="datetime", y="PM2.5", hue="station", ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel("Tanggal")
    plt.ylabel("Konsentrasi PM2.5 (µg/m³)")
    plt.title("Tren PM2.5 dari Waktu ke Waktu")
    st.pyplot(fig)
else:
    st.warning("Tidak ada data yang tersedia dalam rentang waktu dan stasiun yang dipilih.")
