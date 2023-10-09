# IMPORT LIBRARIES
import pandas as pd
import streamlit as st
import docx2txt
import pdfplumber
import re
import PyPDF2
import pickle as pk
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



st.title('             RESUME CLASSIFICATION    ')
st.markdown('<style>h1{color: Blue;}</style>', unsafe_allow_html=True)
st.subheader(':green[PROJECT DEPLOYMENT]')


# FUNCTIONS
def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization
            
    
    data = pd.read_csv("skills.csv") # reading the csv file
            
    
    skills = list(data.columns.values)# extract values
            
    skillset = []
            
    
    for token in tokens:                 # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)
            
   
    for token in noun_chunks:            # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
      
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        
        for para in doc:
            fullText = fullText + para
            
           
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
             
        for paragraph in page_content:
            fullText =  fullText + paragraph
         
    return (fullText)


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile Name','Skills',])
filename = []
predict_name = []
skills = []
resumes =[]


model = pk.load(open("resume_model.pkl", 'rb'))
Vectorizer = pk.load(open("resume_vector.pkl", 'rb'))

upload_file = st.file_uploader('Upload Your Resumes',
                                type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned=preprocess(display(doc_file))
        resumes.append(cleaned)
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        if prediction==0:
            predict_name.append('Peoplesoft resume')
        elif prediction==1:
            predict_name.append('React JS resume')
        elif prediction==2:
            predict_name.append('SQL Developer resume')
        else:
            predict_name.append('Workday resume')
    
        
if len(predict_name) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile Name'] = predict_name
    st.table(file_type.style.format())
    st.header(':blue[WORDCLOUDS]')
    
for x in range(len(resumes)):
    word_cloud = WordCloud(width=1000, height=800, random_state=10, background_color="black", 
                       colormap="Pastel1", collocations=False, stopwords=STOPWORDS).generate(resumes[x])  

    fig=plt.figure(figsize=(10,8), dpi=500)                     
    plt.title('Most Common Words in each resumes', fontsize= 16, fontweight= 'bold')
    plt.imshow(word_cloud)
    plt.axis("off")
    st.subheader(f' :green[{filename[x]}]')
    st.pyplot(fig)
    
    
   