import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import date
import os
import matplotlib.pyplot as plt

# Load model and scalers
model = load_model("lstm_co2_predictor.h5", compile=False)
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

DATA_FILE = "user_data.csv"
TIME_STEPS = 14

# Emission factors
CO2_PER_LITER_DIESEL = 2.68      # kg CO₂ per liter diesel
CO2_PER_TONNE_COAL = 1.9         # kg CO₂ per tonne coal mined (example value)

# Initialize CSV if it doesn't exist
if not os.path.exists(DATA_FILE):
    df_init = pd.DataFrame(columns=["Date", "Fuel_Used_Liters", "Coal_Mined_Tonnes"])
    df_init.to_csv(DATA_FILE, index=False)

# Load user-entered data
df = pd.read_csv(DATA_FILE)

# App Title
st.title("Minelytics : The Intelligent Emission Analytics")

# 1. Daily Input Form
with st.form("daily_input"):
    st.subheader("Enter Today’s Data")
    fuel = st.number_input("Fuel Used (liters)", min_value=500.0, max_value=50000.0, step=100.0)
    coal = st.number_input("Coal Mined (tonnes)", min_value=1000.0, max_value=100000.0, step=500.0)
    submitted = st.form_submit_button("Submit")

if submitted:
    today = date.today().strftime("%Y-%m-%d")
    new_row = pd.DataFrame([{ "Date": today, "Fuel_Used_Liters": fuel, "Coal_Mined_Tonnes": coal }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success(f"✅ Data for {today} saved!")

# 2. Edit/Delete Last 30 Rows
st.subheader("📝 Edit or Delete Records")
editable_rows = df.tail(30).copy()
edited = st.data_editor(
    editable_rows,
    use_container_width=True,
    num_rows="dynamic",
    key="editor"
)

if st.button("💾 Save Changes"):
    df = df.drop(editable_rows.index)  # safer drop
    df = pd.concat([df, edited], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.success("✅ Edits and deletions saved successfully!")

# 3. Prediction Section
if len(df) >= TIME_STEPS:
    st.subheader("📈 Predicted CO₂ Emission for Tomorrow")

    last_14 = df[["Fuel_Used_Liters", "Coal_Mined_Tonnes"]].tail(TIME_STEPS).values
    X_input = scaler_x.transform(last_14).reshape(1, TIME_STEPS, 2)
    y_pred_scaled = model.predict(X_input)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

    st.metric(label="CO₂ Emitted (kg)", value=f"{y_pred:,.2f}")

# 4. Trend & Summary Analysis
if len(df) >= TIME_STEPS:
    st.subheader("📊 CO₂ Emission Analysis & Trend")

    actual_vals = []
    predicted_vals = []
    days_index = []

    for i in range(TIME_STEPS, len(df)):
        seq = df[["Fuel_Used_Liters", "Coal_Mined_Tonnes"]].iloc[i - TIME_STEPS:i].values
        seq_scaled = scaler_x.transform(seq).reshape(1, TIME_STEPS, 2)
        pred_scaled = model.predict(seq_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        predicted_vals.append(pred)

        # actual emission
        fuel_co2 = df["Fuel_Used_Liters"].iloc[i] * CO2_PER_LITER_DIESEL
        coal_co2 = df["Coal_Mined_Tonnes"].iloc[i] * CO2_PER_TONNE_COAL
        actual = fuel_co2 + coal_co2
        actual_vals.append(actual)

        days_index.append(df["Date"].iloc[i])   # keep correct date labels

    # --- Take last 30 days ---
    recent_actual = actual_vals[-30:]
    recent_predicted = predicted_vals[-30:]
    recent_days = days_index[-30:]

    # --- Summary Stats ---
    st.markdown("### 📈 Last 30 Days CO₂ Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Actual (kg)", f"{float(np.mean(recent_actual)) :,.2f}")
    col2.metric("Avg Predicted (kg)", f"{float(np.mean(recent_predicted)) :,.2f}")
    col3.metric("Total Actual (kg)", f"{float(np.sum(recent_actual)) :,.1f}")

    # --- Plot Graph with Date axis ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent_days, recent_actual, label="Actual (Fuel + Coal)", linestyle="--", marker="o")
    ax.plot(recent_days, recent_predicted, label="Predicted (LSTM)", linewidth=2, marker="x")
    ax.set_xlabel("Date")
    ax.set_ylabel("CO₂ Emitted (kg)")
    ax.set_title("CO₂ Emission Trend – Last 30 Days")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)
