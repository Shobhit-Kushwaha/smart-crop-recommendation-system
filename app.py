from pathlib import Path
import json, joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Smart Crop Advisor",page_icon="🌱",layout="wide")
st.title("🌱 Smart Crop Recommendation System")
st.caption("Machine-learning mini project | Soil nutrients + weather → crop recommendation")

model_path=ROOT/"models/crop_model.joblib"
meta_path=ROOT/"models/metadata.json"
data_path=ROOT/"data/Crop_recommendation.csv"
metrics_path=ROOT/"reports/model_comparison.csv"
if not model_path.exists() or not meta_path.exists():
    st.warning("Model not trained yet. Run `python train_model.py` after adding the dataset.")
    st.stop()
model=joblib.load(model_path); meta=json.loads(meta_path.read_text())
df=pd.read_csv(data_path)
metrics=pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

tab1,tab2,tab3,tab4=st.tabs(["Crop Advisor","Dataset Explorer","Model Performance","Project Guide"])
with tab1:
    st.subheader("Enter field and weather conditions")
    st.write("Enter soil nutrient values (N, P, K in mg/kg), temperature (°C), humidity (%), pH and rainfall (mm).")
    with st.form("crop_form"):
        c1,c2,c3=st.columns(3)
        with c1:
            n=st.number_input("Nitrogen (N)",0.0,200.0,50.0,1.0)
            p=st.number_input("Phosphorus (P)",0.0,200.0,50.0,1.0)
            k=st.number_input("Potassium (K)",0.0,250.0,50.0,1.0)
        with c2:
            temp=st.number_input("Temperature (°C)",-5.0,60.0,25.0,.5)
            humidity=st.number_input("Humidity (%)",0.0,100.0,70.0,1.0)
        with c3:
            ph=st.number_input("Soil pH",0.0,14.0,6.5,.1)
            rainfall=st.number_input("Rainfall (mm)",0.0,5000.0,100.0,5.0)
        submitted=st.form_submit_button("Recommend crop",type="primary",use_container_width=True)
    if submitted:
        x=pd.DataFrame([[n,p,k,temp,humidity,ph,rainfall]],columns=meta["features"])
        prediction=model.predict(x)[0]
        st.success(f"Recommended crop according to the model: **{prediction.title()}**")
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(x)[0]
            prob_df=pd.DataFrame({"Crop":model.classes_,"Model score":probs}).sort_values("Model score",ascending=False).head(5)
            st.subheader("Top 5 model suggestions")
            st.dataframe(prob_df.style.format({"Model score":"{:.1%}"}),use_container_width=True)
            st.caption("Scores are model probabilities, not guarantees of crop yield or suitability.")
        st.info("Before planting, confirm results with local agricultural experts and current soil testing, irrigation, season, seed and market conditions.")
with tab2:
    st.subheader("Dataset overview")
    a,b,c=st.columns(3)
    a.metric("Records",f"{len(df):,}")
    b.metric("Crop classes",df["label"].nunique())
    c.metric("Input variables",7)
    st.write("Crop distribution")
    counts=df["label"].value_counts().rename_axis("crop").reset_index(name="records")
    st.plotly_chart(px.bar(counts,x="crop",y="records",title="Records per crop"),use_container_width=True)
    st.dataframe(df.head(20),use_container_width=True)
with tab3:
    st.subheader("Algorithm comparison")
    st.metric("Selected model",meta["best_model"])
    st.metric("Held-out test accuracy",f"{meta['test_accuracy']:.2%}")
    if not metrics.empty:
        st.plotly_chart(px.bar(metrics,x="model",y="accuracy",title="Test accuracy by algorithm",text_auto=".1%"),use_container_width=True)
        st.dataframe(metrics.style.format({"accuracy":"{:.2%}","macro_f1":"{:.3f}"}),use_container_width=True)
    st.caption("A random hold-out score does not establish performance across all farms, seasons, regions or soil types.")
with tab4:
    st.subheader("Project workflow")
    st.markdown("""
    1. Collect soil and weather features from the labelled crop dataset.
    2. Clean data and separate input features from crop labels.
    3. Use a stratified 80:20 train-test split.
    4. Train Random Forest, Extra Trees, KNN and Gradient Boosting.
    5. Compare accuracy and macro-F1 on the same unseen test split.
    6. Save the selected model and serve it through Streamlit.
    """)
    st.write("Model selected:",meta["best_model"])
    st.warning("Educational prototype only. The benchmark dataset may not represent local Indian agro-climatic conditions. Do not treat the prediction as agronomic advice.")
