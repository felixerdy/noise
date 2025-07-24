from datetime import timezone
import pandas as pd
import os
from datetime import datetime, timedelta
import pyarrow as pa
import matplotlib.pyplot as plt
import logging
import requests
import numpy as np


import pyarrow.parquet as pq

PARQUET_FILE = "leq_data.parquet"
BOX_ID = "682b3a8f7f7c2000081caa66"
SENSOR_ID = "682b3a8f7f7c2000081caa69"
BASE_URL = f"https://api.opensensemap.org/boxes/{BOX_ID}/data/{SENSOR_ID}"
threshold = 65


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')


def fetch_weekly_data(start, end):
    url = f"{BASE_URL}?from-date={start.strftime('%Y-%m-%dT%H:%M:%SZ')}&to-date={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    logging.info(f"Fetching data from {url}")
    logging.info(f"Fetching data from {start} to {end}")
    df = pd.read_json(url)
    if not df.empty:
        df["createdAt"] = pd.to_datetime(df["createdAt"])
        df = df.sort_values("createdAt")
        logging.info(f"Fetched {len(df)} records.")
    else:
        logging.info("No data fetched for this period.")
    return df

# Function to check rain for a given interval


def was_raining(start_dt, end_dt, lat=52, lon=7.6):
    date_str = start_dt.strftime('%Y-%m-%d')
    url = f"https://api.brightsky.dev/weather?lat={lat}&lon={lon}&date={date_str}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("weather", [])
        for entry in data:
            ts = pd.to_datetime(entry["timestamp"])
            if start_dt <= ts <= end_dt:
                if entry.get("precipitation", 0) > 0 or "rain" in str(entry.get("condition", "")).lower():
                    return True
        return False
    except Exception as e:
        logging.warning(f"Weather API error for {date_str}: {e}")
        return None

# Helper to extract mean lat/lon for an interval


def get_mean_lat_lon(df, start, end):
    df_interval = df[(df.index >= start) & (df.index <= end)]
    if "location" in df_interval.columns and not df_interval.empty:
        # Convert the column to a NumPy array for easy processing
        locations = np.array(df_interval["location"].tolist())

        # Calculate mean longitude and latitude
        mean_longitude = locations[:, 0].mean()
        mean_latitude = locations[:, 1].mean()

        # round to 6 decimal places
        mean_longitude = round(mean_longitude, 6)
        mean_latitude = round(mean_latitude, 6)

        
        if np.isnan(mean_latitude) or np.isnan(mean_longitude):
            return 52, 7.6
        return mean_latitude, mean_longitude
    return 52, 7.6  # fallback


# Determine start date
if os.path.exists(PARQUET_FILE):
    table = pq.read_table(PARQUET_FILE)
    df_local = table.to_pandas()
    last_date = df_local["createdAt"].max()
    start_date = last_date + timedelta(seconds=1)
else:
    df_local = pd.DataFrame()
    start_date = datetime(2025, 1, 1)

# Download new data week by week

now = datetime.now(timezone.utc)
dfs = []
while start_date.replace(tzinfo=timezone.utc) < now:
    end_date = min(start_date.replace(
        tzinfo=timezone.utc) + timedelta(days=7), now)
    logging.info(
        f"Downloading data for week: {start_date.date()} to {end_date.date()}")
    df_new = fetch_weekly_data(start_date, end_date)
    if not df_new.empty:
        dfs.append(df_new)
        logging.info(f"Appended {len(df_new)} new records.")
    else:
        logging.info("No new records to append.")
    start_date = end_date + timedelta(seconds=1)

# Combine and save
if dfs:
    logging.info(
        f"Saving {sum(len(df) for df in dfs)} new records to {PARQUET_FILE}")
    df_all = pd.concat([df_local] + dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(
        subset="createdAt").sort_values("createdAt")
    table = pa.Table.from_pandas(df_all)
    pq.write_table(table, PARQUET_FILE)
    logging.info("Data saved successfully.")
else:
    df_all = df_local
    logging.info("No new data to save.")

df = df_all.set_index("createdAt")

# # Gleitender Mittelwert (10 Minuten Fenster, falls Messung z.B. alle 60s)
df["leq_rolling"] = df["value"].rolling(
    window="10min",  # 10 Minuten Fenster
    min_periods=1,  # Mindestens 1 Messung im Fenster
    center=True  # Gleitender Mittelwert zentriert auf den Zeitstempel
).mean()

# # Zu laut: Grenzwert überschritten (z. B. 55 dB)
df["too_loud"] = df["leq_rolling"] > threshold

# # Gruppieren: zusammenhängende 'zu laut'-Phasen
df["group"] = (df["too_loud"] != df["too_loud"].shift()).cumsum()
loud_groups = df[df["too_loud"]].groupby("group")

# # Intervall-Tabelle erzeugen
intervals = loud_groups.apply(
    lambda x: pd.Series({
        "start": x.index[0],
        "end": x.index[-1],
        "duration_min": (x.index[-1] - x.index[0]).total_seconds() / 60
    }),
    include_groups=False
).reset_index(drop=True)

# # Nur Intervalle, die länger als 5 Minuten zu laut waren
min_duration = 5
intervals = intervals[intervals["duration_min"] >= min_duration]

plt.figure(figsize=(12, 6))

# Plot 10min rolling mean (original)
plt.plot(df.index, df["leq_rolling"],
         label="Gleitender Mittelwert (10 Min)", color='blue')

# Calculate and plot 24h rolling mean with improved color visibility
df["leq_rolling_24h"] = df["value"].rolling(
    window="24h",
    min_periods=1,
    center=True
).mean()
plt.plot(df.index, df["leq_rolling_24h"],
         label="Gleitender Mittelwert (24h)", color='black')

plt.axhline(y=threshold, color='red', linestyle='--',
            label=f"Schwellenwert ({threshold} dB)")
for _, row in intervals.iterrows():
    plt.axvspan(row["start"], row["end"], color='orange', alpha=0.5, label='Zu laut')

# Highlight weekends as a flat bar at the bottom
ymin, ymax = plt.ylim()
weekend_y = ymin + 0.01 * (ymax - ymin)
weekend_height = 0.03 * (ymax - ymin)
dates = pd.date_range(df.index.min().date(), df.index.max().date(), freq='D')
for date in dates:
    if date.weekday() >= 5:  # Saturday or Sunday
        start = pd.Timestamp(date).replace(tzinfo=df.index[0].tz)
        end = start + pd.Timedelta(days=1)
        plt.axvspan(start, end, ymin=0, ymax=0.03, color='gray', alpha=0.4, label='Wochenende')


# Markiere zu laute Intervalle mit Regen in Blau
# Wir prüfen und speichern das Regen-Ergebnis für jedes Intervall, um Dopplungen und API-Calls zu vermeiden
intervals["raining"] = [
    was_raining(
        row["start"],
        row["end"],
        *get_mean_lat_lon(df, row["start"], row["end"])
    ) for _, row in intervals.iterrows()
]
for _, row in intervals.iterrows():
    if row["raining"]:
        plt.axvspan(row["start"], row["end"], color='blue',
                    alpha=0.3, label='Zu laut & Regen')

plt.title("Lärmpegelanalyse")
plt.xlabel("Zeit")
plt.ylabel("Lärmpegel (dB)")

# Nur einen Eintrag für die Legende
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.grid()
plt.tight_layout()
plt.savefig("noise_analysis.png")

# # Ausgabe in Textform
for _, row in intervals.iterrows():
    start = row["start"]
    end = row["end"]
    duration = int(row["duration_min"])
    raining = row["raining"]
    rain_str = " (Regen)" if raining else (
        " (kein Regen)" if raining is False else " (Wetterdaten nicht verfügbar)")
    print(f"Von {start.strftime('%Y-%m-%d %H:%M')} bis {end.strftime('%Y-%m-%d %H:%M')} war es {duration} Minuten zu laut.{rain_str}")

# Generate interactive HTML report
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["leq_rolling"],
    mode='lines',
    name='Gleitender Mittelwert (10 Min)',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["leq_rolling_24h"],
    mode='lines',
    name='Gleitender Mittelwert (24h)',
    line=dict(color='black')
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=[threshold] * len(df),
    mode='lines',
    name=f'Schwellenwert ({threshold} dB)',
    line=dict(color='red', dash='dash')
))
for _, row in intervals.iterrows():
    fig.add_vrect(
        x0=row["start"],
        x1=row["end"],
        fillcolor='orange',
        opacity=0.5,
        line_width=0,
        name='Zu laut'
    )
    if row["raining"]:
        fig.add_vrect(
            x0=row["start"],
            x1=row["end"],
            fillcolor='blue',
            opacity=0.3,
            line_width=0,
            name='Zu laut & Regen'
        )
fig.update_layout(
    title='Lärmpegelanalyse',
    xaxis_title='Zeit',
    yaxis_title='Lärmpegel (dB)',
    legend=dict(x=0, y=1, traceorder='normal'),
    template='plotly_white'
)
# Enable zoom with scrollwheel
fig.write_html("index.html", config={"scrollZoom": True})