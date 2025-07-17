import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load model and test data
model = joblib.load("C:/Users/diyag/OneDrive/Pictures/Desktop/fraud_detection_with_shap/xgb_model.pkl")
X_test = pd.read_csv("C:/Users/diyag/OneDrive/Pictures/Desktop/fraud_detection_with_shap/X_test.csv")

# Create SHAP explainer (use TreeExplainer for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

#  SHAP summary plot (feature importance)
shap.plots.beeswarm(shap_values)

#  SHAP force plot (single transaction)
shap.initjs()
shap.force_plot(
    base_value=explainer.expected_value,
    shap_values=shap_values.values[0],
    features=X_test.iloc[0]
)
