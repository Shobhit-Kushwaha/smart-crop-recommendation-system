# Viva Questions — Smart Crop Recommendation

1. **What is the objective?** Predict a crop category from soil and weather inputs in a benchmark dataset.
2. **What type of ML problem is this?** Multi-class supervised classification.
3. **What is the target?** The crop label.
4. **What are N, P and K?** Nitrogen, phosphorus and potassium.
5. **Why is pH important?** It indicates soil acidity/alkalinity and affects nutrient availability.
6. **What is rainfall measured in?** Millimetres in the dataset.
7. **What is humidity?** Amount of water vapour in air, represented as a percentage here.
8. **What is a feature?** An input variable used by the model.
9. **What is a label?** The known output class used during supervised training.
10. **Why use stratified splitting?** To preserve approximate class proportions in both splits.
11. **What is overfitting?** When a model fits training data too closely and generalizes poorly.
12. **What is Random Forest?** An ensemble that combines predictions from many decision trees.
13. **What is Extra Trees?** A tree ensemble that adds more randomization when selecting splits.
14. **What is KNN?** A model that predicts using nearby training examples.
15. **What is Gradient Boosting?** An ensemble built sequentially to correct previous errors.
16. **What is accuracy?** Correct predictions divided by total predictions.
17. **What is macro-F1?** The unweighted mean of F1-score across classes.
18. **Why compare multiple models?** To select the model that performs best on the same held-out data.
19. **What is a confusion matrix?** A table showing true vs predicted classes.
20. **Does high test accuracy guarantee real farm success?** No; local and seasonal validation is required.
21. **What extra factors affect crop choice?** Season, soil texture, irrigation, seed, pests, market and local climate.
22. **What is Streamlit?** A Python framework for building interactive data apps.
23. **What is a train-test split?** Separating data for fitting and independent evaluation.
24. **Why is the model not a final farming advisor?** The benchmark dataset and features do not capture every local agronomic condition.
25. **How can it be improved?** Add region-specific data, crop calendars, yield data and external validation.
