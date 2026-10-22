
import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calories Predictor")
st.write("Decision Tree Machine Learning Model")
st.write("Enter drink details to predict calories.")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "starbucks_model.pkl"
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Upload starbucks_model.pkl "
        "to the same GitHub repository as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.subheader("Enter Beverage Information")

# These inputs must match the feature columns
# used when training your model.
feature_names = list(model.feature_names_in_) if hasattr(
    model, "feature_names_in_"
) else None

if not feature_names:
    st.error(
        "Feature names are unavailable. Please save the model "
        "together with the training feature columns."
    )
    st.stop()

inputs = {}

for col in feature_names:
    if pd.api.types.is_numeric_dtype(
        getattr(model, "feature_names_in_", [])
    ):
        pass

    # Numeric fields
    if col.lower() in [
        "caffeine (mg)", "caffeine", "fat (g)",
        "carb. (g)", "carbs", "protein (g)",
        "sugars (g)", "sodium (mg)", "fiber (g)"
    ]:
        inputs[col] = st.number_input(
            col, min_value=0.0, value=0.0
        )
    else:
        inputs[col] = st.text_input(
            col, value=""
        )

if st.button("Predict Calories"):
    input_df = pd.DataFrame([inputs], columns=feature_names)

    # Convert numeric training features to numbers
    for col in feature_names:
        if col in input_df.columns:
            input_df[col] = input_df[col].replace("", "0")

    try:
        prediction = model.predict(input_df)[0]
        st.success(
            f"Predicted Calories: {prediction:.2f} kcal"
        )
    except Exception as e:
        st.error(
            "Prediction failed. Check that the inputs match "
            f"the model's training features. Details: {e}"
        )

st.caption("Educational project — predictions depend on training data.")

