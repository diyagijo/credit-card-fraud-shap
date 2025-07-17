import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set up Streamlit page
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title(" Credit Card Fraud Detection with SHAP Explainability")
st.markdown("Built using **XGBoost** and **SHAP** — audit-compliant model interpretability for finance.")

# Load trained model
model = joblib.load("models/xgb_model.pkl")


# Upload test CSV
uploaded_file = st.file_uploader("Upload your test CSV file (X_test.csv)", type=["csv"])

if uploaded_file is not None:
    X = pd.read_csv(uploaded_file)
    st.subheader(" Preview of Uploaded Data")
    st.dataframe(X.head())

    # Predict fraud
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    X["Predicted Fraud"] = preds
    X["Fraud Probability"] = probs

    # Show fraudulent transactions
    st.subheader(" Detected Fraudulent Transactions")
    frauds = X[X["Predicted Fraud"] == 1]
    st.write(frauds)

    # Prepare input for SHAP
    drop_cols = ['Predicted Fraud', 'Fraud Probability']
    X_input = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')

    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_input)

    # SHAP summary plot
    st.subheader(" SHAP Summary Plot")
    fig1, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig1)

    # SHAP force plot (first row)
    st.subheader("SHAP Force Plot (First Transaction)")
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values.values[0],
        features=X_input.iloc[0],
        matplotlib=False,
        show=False
    )

    # Save and embed force plot
    shap.save_html("force_plot.html", force_plot)
    with open("force_plot.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=400)

else:
    st.info("Please upload your `X_test.csv` to begin.")
