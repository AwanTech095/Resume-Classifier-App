# Resume Classifier App
This is a Machine Learning-powered web application built with **Streamlit** that classifies resumes into different job roles such as `Data Science`, `Web Designing`, `HR`, and more based on their content.

---

## Features
- Cleaned and preprocessed real-world resume data
- TF-IDF vectorization with unigrams and bigrams
- Multi-class classification using Logistic Regression
- Fast and intuitive web UI built with Streamlit
- Deployed on Streamlit Cloud

---

## How It Works
1. User pastes a resume.
2. App vectorizes it using the trained `TF-IDF` model.
3. Model predicts the job category.
4. Result is displayed with a clean UI.

---

## Tech Stack
- Python
- Pandas, NLTK, spaCy
- Scikit-learn (Logistic Regression)
- Streamlit
- Joblib (model persistence)

---

## Run Locally

```bash
git clone https://github.com/AwanTech095/Resume-Classifier-App.git
cd Resume-Classifier-App
pip install -r requirements.txt
streamlit run app.py
