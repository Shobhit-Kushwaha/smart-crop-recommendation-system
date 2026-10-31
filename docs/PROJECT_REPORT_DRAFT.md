# Smart Crop Recommendation System Using Machine Learning

## Abstract
This project builds a supervised classification system that recommends a crop category from soil nutrient and weather measurements. It uses nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH and rainfall as input features. Four algorithms are compared using a stratified hold-out test set. A Streamlit interface accepts user-entered values and displays a predicted crop and top model scores. The prototype is intended for academic learning and requires local agronomic validation before any practical use.

## Problem statement
Farmers and agricultural learners may need to interpret several soil and climate variables together. This project demonstrates how a classification model can learn relationships between these variables and crop labels in a benchmark dataset.

## Dataset
Use the public Crop Recommendation dataset (CSV source and citation details should be verified by the student). Required columns: N, P, K, temperature, humidity, ph, rainfall, label. Document the actual dataset source, version, license and record count used.

## Methodology
1. Inspect and clean missing records.
2. Separate seven numerical predictors and the crop label.
3. Split into training and testing sets with stratification.
4. Train Random Forest, Extra Trees, KNN and Gradient Boosting.
5. Compare held-out accuracy and macro-F1.
6. Save the best estimator and create a Streamlit prediction interface.

## Results
Copy the actual values from `reports/model_comparison.csv`. Do not invent or pre-fill an accuracy claim. Explain that the score is tied to this dataset and split.

## Limitations and future scope
The dataset may not represent local farms, all seasons or future climate patterns. Future work can integrate location-specific soil tests, seasonal data, regional crop calendars, irrigation constraints, yield and market prices, and external validation with agricultural experts.
