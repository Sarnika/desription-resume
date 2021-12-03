import docx2txt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pdfplumber
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

padding = 0
st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Job Description - Resume Similarity")
col1,col2 = st.columns(2)
resume = col1.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_description = col2.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])

if st.button("Run"):

    if resume is not None:
        if resume.type == "text/plain":
            # Read as string (decode bytes to string)
            raw_text = str(resume.read(), "utf-8")
            resume = raw_text

        elif resume.type == "application/pdf":
            try:
                with pdfplumber.open(resume) as pdf:
                    pages = pdf.pages[0]
                    resume = pages.extract_text()
            except:
                st.write("None")

    else:
        resume = docx2txt.process(resume)


    if job_description is not None:
        if job_description.type == "text/plain":
            # Read as string (decode bytes to string)
            job_description = str(job_description.read(), "utf-8")

        elif job_description.type == "application/pdf":
            try:
                with pdfplumber.open(job_description) as pdf:
                    pages = pdf.pages[0]
                    job_description = pages.extract_text()
            except:
                st.write("None")

    else:
        job_description = docx2txt.process(job_description)



    text = [resume, job_description]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2) # round to two decimal
    st.header("Resume matches about "+ str(matchPercentage)+ "% of the job description.")

    st.header("Word Cloud representation of the job description.")
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          width=1200,
                          height=1000
                          ).generate(job_description)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    fig = plt.show()
    st.pyplot(fig)