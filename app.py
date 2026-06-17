import gradio as gr
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Load saved files
# =========================
model = joblib.load("best_model.pkl")
feature_info = joblib.load("feature_info.pkl")
explainer = joblib.load("shap_explainer.pkl")

numeric_features = feature_info["numeric_features"]
categorical_features = feature_info["categorical_features"]
all_features = numeric_features + categorical_features


# =========================
# SHAP VISUALIZATION (BAR + PIE)
# =========================

def create_shap_plot(explanation):
    names = list(explanation.keys())
    values = list(explanation.values())

    sorted_items = sorted(
        zip(names, values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ================= BAR =================
    colors = ['red' if v > 0 else 'green' for v in values]
    ax1.barh(names, values, color=colors)
    ax1.axvline(0, color='black')
    ax1.set_title("SHAP Feature Impact", pad=10)
    ax1.set_xlabel("Impact")

    # ================= PIE =================
    abs_values = np.abs(values)
    total = np.sum(abs_values)
    percentages = (abs_values / total) * 100 if total != 0 else abs_values

    ax2.pie(
        percentages,
        labels=names,
        autopct='%1.1f%%',
        startangle=140,
        textprops={'fontsize': 9}
    )
    ax2.set_title("Feature Contribution (%)", pad=10)

    
    fig.suptitle("SHAP Analysis (Bar + Pie)", fontsize=18, fontweight='bold', y=0.98)

    
    plt.subplots_adjust(top=0.75, wspace=0.3)

    return fig


# =========================
# Prediction Function
# =========================
def predict(age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal):

    input_data = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]], columns=all_features)

    # prediction
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # =========================
    # SHAP
    # =========================
    X_transformed = model.named_steps["preprocessor"].transform(input_data)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = np.array(shap_vals).flatten()

    explanation = dict(zip(all_features, shap_vals))

    # =========================
    # TEXT EXPLANATION
    # =========================
    sorted_features = sorted(
        explanation.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    text = "🔥 Top Contributing Factors:\n\n"

    for feature, value in sorted_features[:5]:
        if value > 0:
            text += f"🔴 {feature} increased risk by {abs(value):.4f}\n"
        else:
            text += f"🟢 {feature} decreased risk by {abs(value):.4f}\n"

    # =========================
    # RESULT
    # =========================
    label = "🫀 Heart Disease Detected" if pred == 1 else "💚 No Heart Disease"
    probability = f"{prob*100:.2f}%"

    fig = create_shap_plot(explanation)

    return label, probability, text, fig


# =========================
# UI
# =========================
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="pink",
        neutral_hue="slate"
    ),
    title="Heart Disease Predictor"
) as app:

    gr.Markdown("""
    # 🫀 Heart Disease Prediction System
   
   Predict heart disease risk using Machine Learning and SHAP Explainability.
    """)

    with gr.Row():
        age = gr.Number(label="Age")
        sex = gr.Radio([0, 1], label="Sex")

    with gr.Row():
        cp = gr.Number(label="Chest Pain Type")
        trestbps = gr.Number(label="Resting BP")
        chol = gr.Number(label="Cholesterol")

    with gr.Row():
        fbs = gr.Radio([0, 1], label="FBS")
        restecg = gr.Number(label="Rest ECG")
        thalach = gr.Number(label="Max Heart Rate")

    with gr.Row():
        exang = gr.Radio([0, 1], label="Exercise Angina")
        oldpeak = gr.Number(label="Oldpeak")
        slope = gr.Number(label="Slope")

    with gr.Row():
        ca = gr.Number(label="Vessels (ca)")
        thal = gr.Number(label="Thal")

    btn = gr.Button("🔍 Predict", variant="primary")

    result = gr.Textbox(label="Prediction")
    probability = gr.Textbox(label="Risk Probability")
    explanation = gr.Textbox(label="Explanation", lines=8)
    shap_plot = gr.Plot(label="SHAP Analysis (Bar + Pie)")

    btn.click(
        fn=predict,
        inputs=[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach,
            exang, oldpeak, slope,
            ca, thal
        ],
        outputs=[
            result,
            probability,
            explanation,
            shap_plot
        ]
    )

app.launch()