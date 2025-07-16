import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('resume_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Page configuration
st.set_page_config(page_title="Resume Classifier", page_icon="💼", layout="centered")

# --- Title & Subheading ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>💼 Smart Resume Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict the professional field of any resume using AI</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Text Input Area ---
resume_text = st.text_area("📄 Paste your resume text below:", height=300, help="Include relevant experience, education, skills, and projects.")

# --- Predict Button ---
if st.button("🔍 Classify Resume"):
    if resume_text.strip() == "":
        st.warning("⚠️ Please enter some resume text first.")
    else:
        with st.spinner("Analyzing resume..."):
            X_input = vectorizer.transform([resume_text])
            predicted_category = model.predict(X_input)[0]

        # --- Result Display (Updated styling) ---
        st.markdown(
            f"""
            <div style='
                background-color: #1e1e1e;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
            '>
                <h4 style='color: #FFFFFF; text-align: center;'>🎯 Predicted Resume: <b>{predicted_category}</b></h4>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Sidebar ---
st.sidebar.title("📘 About")
st.sidebar.info(
    "This web app uses a machine learning model (TF-IDF + Logistic Regression) to classify resumes into job categories like:\n\n"
    "- Data Science\n- HR\n- Web Designing\n- DevOps\n- Java Developer\n- and more!"
)

st.sidebar.title("🛠 Tech Stack")
st.sidebar.success("Python · Scikit-learn · Pandas · Numpy · NLTK · SpaCy · Streamlit")

st.sidebar.markdown("---")
st.sidebar.markdown("🧑‍💻 Created by **Rabbi Awan**")


