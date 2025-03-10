import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from uploaded PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description with resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarity_scores = cosine_similarity([job_desc_vector], resume_vectors).flatten()
    
    return similarity_scores

# Streamlit UI
def main():
    st.title("AI Resume Screening & Ranking System")
    
    job_desc = st.text_area("Enter Job Description:")
    uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Rank Resumes"):
        if not job_desc:
            st.warning("Please enter a job description.")
            return
        
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
            return
        
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_desc, resumes_text)
        
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        
        st.write("### Ranked Resumes:")
        st.dataframe(results)

if __name__ == "__main__":
    main()
