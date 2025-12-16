import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import date
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Minelytics",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Minelytics : Intelligent Emission & Soil Analytics")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2 = st.tabs(["🌱 Soil Type Classifier", "📈 CO₂ Emission Predictor"])

# ==================================================
# TAB 1 : CNN SOIL CLASSIFIER
# ==================================================
with tab1:
    st.header("🌱 Soil Type Classification using CNN")

    # Load CNN model
    cnn_model = load_model("soil_classifier_model.h5")

    class_labels = [
        "Alluvial soil",
        "Black soil",
        "Clay soil",
        "Laterite soil",
        "Red soil"
    ]

    uploaded_file = st.file_uploader(
        "Upload a soil image...",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(128, 128))
        st.image(img, caption="Uploaded Soil Image", use_column_width=True)

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction[0])]

        st.success(f"✅ Predicted Soil Type: **{predicted_class}**")

        # Plant suggestions
        plant_suggestions = {
            "Alluvial soil": ["Bamboo", "Sunflower", "Mint", "Spinach", "Lemongrass"],
            "Black soil": ["Cotton", "Guava", "Castor", "Jatropha", "Marigold"],
            "Clay soil": ["Willow", "Rice", "Bamboo", "Canna Lily", "Coleus"],
            "Laterite soil": ["Cashew", "Coconut", "Eucalyptus", "Pineapple", "Lemongrass"],
            "Red soil": ["Millet", "Pigeon Pea", "Eucalyptus", "Teak", "Sorghum"]
        }

        st.subheader("🌿 Suggested Fast-Growing & CO₂-Absorbing Plants")
        for plant in plant_suggestions[predicted_class]:
            st.write(f"• {plant}")

# ==================================================
# TAB 2 : LSTM CO₂ EMISSION PREDICTOR
# ==================================================
with tab2:
    st.header("📈 CO₂ Emission Prediction using LSTM")

    # Load models & scalers
    lstm_model = load_model("lstm_co2_predictor.h5", compile=False)
    scaler_x = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    DATA_FILE = "user_data.csv"
    TIME_STEPS = 14

    CO2_PER_LITER_DIESEL = 2.68
    CO2_PER_TONNE_COAL = 1.9

    # Initialize CSV
    if not os.path.exists(DATA_FILE):
        df_init = pd.DataFrame(
            columns=["Date", "Fuel_Used_Liters", "Coal_Mined_Tonnes"]
        )
        df_init.to_csv(DATA_FILE, index=False)

    df = pd.read_csv(DATA_FILE)

    # ----------------------------
    # Input Form
    # ----------------------------
    with st.form("daily_input"):
        st.subheader("📝 Enter Today’s Data")
        fuel = st.number_input(
            "Fuel Used (liters)",
            min_value=500.0,
            max_value=50000.0,
            step=100.0
        )
        coal = st.number_input(
            "Coal Mined (tonnes)",
            min_value=1000.0,
            max_value=100000.0,
            step=500.0
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        today = date.today().strftime("%Y-%m-%d")
        new_row = pd.DataFrame([{
            "Date": today,
            "Fuel_Used_Liters": fuel,
            "Coal_Mined_Tonnes": coal
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success(f"✅ Data for {today} saved successfully!")

    # ----------------------------
    # Edit / Delete
    # ----------------------------
    st.subheader("✏️ Edit or Delete Last 30 Records")

    editable_rows = df.tail(30).copy()
    edited = st.data_editor(
        editable_rows,
        use_container_width=True,
        num_rows="dynamic"
    )

    if st.button("💾 Save Changes"):
        df = df.drop(editable_rows.index)
        df = pd.concat([df, edited], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ Changes saved!")

    # ----------------------------
    # Prediction
    # ----------------------------
    if len(df) >= TIME_STEPS:
        st.subheader("📈 Tomorrow’s Predicted CO₂ Emission")

        last_14 = df[
            ["Fuel_Used_Liters", "Coal_Mined_Tonnes"]
        ].tail(TIME_STEPS).values

        X_input = scaler_x.transform(last_14).reshape(1, TIME_STEPS, 2)
        y_pred_scaled = lstm_model.predict(X_input)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

        st.metric("CO₂ Emitted (kg)", f"{y_pred:,.2f}")

    # ----------------------------
    # Trend Analysis
    # ----------------------------
    if len(df) >= TIME_STEPS:
        st.subheader("📊 CO₂ Emission Trend (Last 30 Days)")

        actual_vals, predicted_vals, days = [], [], []

        for i in range(TIME_STEPS, len(df)):
            seq = df[
                ["Fuel_Used_Liters", "Coal_Mined_Tonnes"]
            ].iloc[i - TIME_STEPS:i].values

            seq_scaled = scaler_x.transform(seq).reshape(1, TIME_STEPS, 2)
            pred_scaled = lstm_model.predict(seq_scaled)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]

            fuel_co2 = df["Fuel_Used_Liters"].iloc[i] * CO2_PER_LITER_DIESEL
            coal_co2 = df["Coal_Mined_Tonnes"].iloc[i] * CO2_PER_TONNE_COAL

            actual_vals.append(fuel_co2 + coal_co2)
            predicted_vals.append(pred)
            days.append(df["Date"].iloc[i])

        recent_actual = actual_vals[-30:]
        recent_pred = predicted_vals[-30:]
        recent_days = days[-30:]

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Actual (kg)", f"{np.mean(recent_actual):,.2f}")
        col2.metric("Avg Predicted (kg)", f"{np.mean(recent_pred):,.2f}")
        col3.metric("Total Actual (kg)", f"{np.sum(recent_actual):,.1f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(recent_days, recent_actual, label="Actual", marker="o", linestyle="--")
        ax.plot(recent_days, recent_pred, label="Predicted", marker="x")
        ax.set_xlabel("Date")
        ax.set_ylabel("CO₂ (kg)")
        ax.set_title("CO₂ Emission Trend")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
