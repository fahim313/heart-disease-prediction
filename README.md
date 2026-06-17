# Heart Disease Prediction System 🫀

This project is a Machine Learning-based web application designed to predict the risk of heart disease based on user health metrics. The application provides not only accurate predictions but also offers explainability using the **SHAP (SHapley Additive exPlanations)** library to visualize the impact of each feature on the final result.

## 🖥️ Live Demo
You can access the live application here:
[Hugging Face Space - Heart Disease Predictor](https://huggingface.co/spaces/250H1M/Heart-Disease-Predictor)

## 📸 Preview
![App UI Preview](https://huggingface.co/spaces/250H1M/Heart-Disease-Predictor/resolve/main/image_a6243d.jpg)

## 🧠 Model Performance
To build this system, I experimented with multiple machine learning algorithms:
* **Random Forest (RF)**
* **Logistic Regression (LR)**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**

After evaluating the performance, the **Random Forest (RF)** model demonstrated the highest accuracy and reliability for this dataset. Therefore, the application uses the Random Forest model as the primary engine for predictions.

## 📓 Technical Notebook
You can explore the full data analysis, preprocessing, and model training process in my Google Colab notebook:
[View Project Notebook](https://colab.research.google.com/drive/1yNciS9wZeU4zBjzW4jHMqkyJlF9WbSaH?usp=sharing)

## 🛠️ Key Features
- **Accurate Prediction:** Optimized prediction using the Random Forest classifier (Achieved 91.30% accuracy).
- **Explainable AI (XAI):** Provides visual insights into how specific health factors increase or decrease risk using SHAP analysis.
- **User-Friendly Interface:** Built with Gradio for an intuitive and interactive user experience.

## 📊 Dataset
The model was trained using the **Cleveland Heart Disease Dataset**.
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)

## 🚀 How to Run Locally
To run this project on your local machine:
1. Clone this repository: `git clone https://huggingface.co/spaces/250H1M/Heart-Disease-Predictor`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Launch the app: `python app.py`

## 🏗️ Tech Stack
- **Language:** Python
- **ML Framework:** Scikit-Learn
- **Web Interface:** Gradio
- **Explainability:** SHAP
- **Deployment:** Hugging Face Spaces

---
*Disclaimer: This application is intended for educational purposes only and should not be considered a substitute for professional medical advice, diagnosis, or treatment.*
