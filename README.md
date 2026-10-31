# Smart Crop Recommendation System

A practical ML mini-project that recommends a crop label from soil nutrient and weather inputs.

## Dataset setup (required)
This project expects the public Crop Recommendation dataset with these columns:
`N, P, K, temperature, humidity, ph, rainfall, label`

Download `Crop_recommendation.csv` from the Kaggle dataset:
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Place the CSV here:
`data/Crop_recommendation.csv`

The dataset is commonly used for educational crop classification. Check its provenance and terms before redistribution or formal publication. Do not claim it represents all Indian farms.

## Run on Windows PowerShell
Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Open `http://localhost:8501`.

If activation is blocked, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Features
- Real user input form for N, P, K, temperature, humidity, pH and rainfall.
- Top-5 crop suggestions with model scores (where supported).
- Dataset explorer and crop distribution chart.
- Comparison of Random Forest, Extra Trees, KNN and Gradient Boosting.
- Saved model, metrics, confusion matrix and classification report.

## Important limitations
The model predicts labels in the training dataset, not guaranteed yield, profit or agronomic suitability. Local crop choice also depends on season, soil texture, irrigation, seed variety, pests, market access and local agronomy. Use a current soil test and local agriculture expert before real planting decisions.
