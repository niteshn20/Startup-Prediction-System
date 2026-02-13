import streamlit as st
import numpy as np
import pickle

# Loading Model

model = pickle.load(open("startup_survival_model.pkl", "rb"))

# UI

st.set_page_config(page_title="Startup Survival Prediction", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    background-image: radial-gradient(circle at 20% 20%, #1f77b4 0%, transparent 20%),
                      radial-gradient(circle at 80% 80%, #ff7f0e 0%, transparent 20%);
}
</style>
""", unsafe_allow_html=True)


st.title("Startup Survival Prediction System")
st.write("Enter startup details to predict whether the startup will survive or fail.")

#User Input

funding_total_usd = st.number_input("Total Funding (USD$)", min_value=0.0, value=1000000.0)
funding_rounds = st.number_input("Number of Funding Rounds", min_value=0, value=1)

category_list = st.number_input("Category Code (Encoded)", min_value=0, value=1000)
country_code = st.number_input("Country Code (Encoded)", min_value=0, value=100)

startup_age = st.number_input("Startup Age (Years)", min_value=0.0, value=2.0)



if st.button("Predict Startup Survival"):

    #Model Input
    input_data = np.array([[funding_total_usd, funding_rounds, category_list, country_code, startup_age]])

    # Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.success(f"Startup WILL SURVIVE (Probability: {prob:.2f})")
    else:
        st.error(f"Startup MAY FAIL (Probability: {1 - prob:.2f})")

# Footer

st.markdown("---")
st.markdown("Startup Survival Prediction")
st.markdown("Developed by Nitesh")
