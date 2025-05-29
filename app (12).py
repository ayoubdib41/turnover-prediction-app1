
import streamlit as st
import pandas as pd
import joblib

# 📦 Chargement du modèle et du DataFrame de référence
model = joblib.load("project_full_pipeline.pkl")
df_values = joblib.load("hr_df_for_streamlit_values.joblib")

st.title("📊 Prédiction du Turn-over chez HumanForYou")
st.markdown("Remplissez les champs pour savoir si l'employé risque de quitter.")

# 🧾 Formulaire dynamique
user_input = {}
for col in df_values.columns:
    if df_values[col].dtype == 'object':
        user_input[col] = st.selectbox(col, df_values[col].dropna().unique())
    else:
        user_input[col] = st.number_input(col, value=float(df_values[col].mean()))

# ▶️ Prédiction
input_df = pd.DataFrame([user_input])
if st.button("🔍 Lancer la prédiction"):
    prediction = model.predict(input_df)[0]
    probas = model.predict_proba(input_df)[0][1]
    st.subheader("🎯 Résultat")
    st.write(f"**Attrition prédite** : {'Oui' if prediction == 1 else 'Non'}")
    st.progress(int(probas * 100))
    st.write(f"📈 Probabilité de départ estimée : {probas:.2%}")
