# Notebook Guide

## Credit Card Fraud Detection

This directory contains the Jupyter Notebooks used throughout
the Credit Card Fraud Detection project.

## Notebook Execution Order

Run the notebooks in the following order:

| Order | Notebook | Purpose |
|---|---|---|
| 01 | 01_problem_understanding.ipynb | Define the business and ML problem |
| 02 | 02_data_understanding.ipynb | Understand dataset structure |
| 03 | 03_data_cleaning.ipynb | Validate and clean the dataset |
| 04 | 04_eda.ipynb | Perform exploratory data analysis |
| 05 | 05_feature_engineering.ipynb | Create additional features |
| 06 | 06_preprocessing.ipynb | Split and preprocess the data |
| 07 | 07_imbalanced_data.ipynb | Investigate sampling techniques |
| 08 | 08_model_building.ipynb | Train baseline models |
| 09 | 09_hyperparameter_tuning.ipynb | Optimize model parameters |
| 10 | 10_model_evaluation.ipynb | Evaluate the final model |
| 11 | 11_shap_explainability.ipynb | Explain model predictions |

## Important Notes

1. Run notebooks from the project root when possible.
2. Keep the raw dataset inside data/raw/.
3. Do not modify the original raw dataset.
4. Execute notebooks in the documented order.
5. Use the same random state and splitting strategy.
6. Never apply SMOTE to validation or test data.
7. Never use test data for hyperparameter tuning.
8. Select the classification threshold using validation data.
9. Evaluate the final model on the test set only after model selection.
10. Save generated figures and metrics in the reports directory.

## Reproducibility

The project uses a fixed random state to improve reproducibility.

The final model is stored in the models directory.

The evaluation results and figures are stored in reports/metrics/
and reports/figures/.