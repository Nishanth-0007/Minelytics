import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("soil_classifier_model.h5")
class_labels = ['Alluvial soil', 'Black soil', 'Clay soil', 'Laterite soil', 'Red soil']

st.title("🌱 Soil Type Classifier")

uploaded_file = st.file_uploader("Upload a soil image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction[0])]

    st.success(f"✅ Predicted Soil Type: {predicted_class}")

    # --- Suggested plants for each soil type ---
    plant_suggestions = {
        "Alluvial soil": ["Bamboo", "Sunflower", "Mint", "Spinach", "Lemongrass"],
        "Black soil": ["Cotton", "Guava", "Castor", "Jatropha", "Marigold"],
        "Clay soil": ["Willow", "Rice", "Bamboo", "Canna Lily", "Coleus"],
        "Laterite soil": ["Cashew", "Coconut", "Eucalyptus", "Pineapple", "Lemon Grass"],
        "Red soil": ["Millet", "Pigeon Pea", "Eucalyptus", "Teak", "Sorghum"]
    }

    if predicted_class in plant_suggestions:
        st.subheader("🌿 Suggested Fast-Growing & CO₂-Absorbing Plants:")
        for plant in plant_suggestions[predicted_class]:
            st.write(f"- {plant}")
