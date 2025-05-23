
import streamlit as st
import joblib
import numpy as np

# Model laden
model = joblib.load("logistic_model_offerte.pkl")

st.title("Offerte Conversie Voorspeller")

st.markdown("""
Voer hier de gegevens van een offerte in om te voorspellen of deze waarschijnlijk geconverteerd wordt.
""")
# Invoervelden
totaal = st.number_input("Totaalbedrag (€)", min_value=0.0, step=100.0)
marge = st.number_input("Marge (€)", min_value=0.0, step=10.0)
marge_pct = st.number_input("Marge (%)", min_value=0.0, max_value=1.0, step=0.01)

verkoper = st.text_input("Verkoper (exacte naam)")
rang = st.text_input("Rang (bijv. a, b, c)")

# Voorspellen
if st.button("Voorspel conversie"):
    input_data = {
        "Totaal": [totaal],
        "Marge": [marge],
        "Marge (%)": [marge_pct],
        "Verkoper": [verkoper],
        "Rang": [rang]
    }
    import pandas as pd
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Voorspelling: Conversie met kans van {prob:.2%}")
    else:
        st.warning(f"Voorspelling: Geen conversie (kans: {prob:.2%})")
