import streamlit as st 
import pickle

pickle_in = open("final_model2.pkl","rb")
classifier=pickle.load(pickle_in)

def predict_model(Income,Kidhome,Teenhome,Cust_Age,Num_Accepted_Cmp,education,marry):
    prediction=classifier.predict([[Income,Kidhome,Teenhome,Cust_Age,Num_Accepted_Cmp,education,marry]])
    print(prediction)
    return prediction


def main():
    st.title('Clustering Customer Segmentation')
    st.header('User Input Parameters')
    Income = st.text_input("Insert the income")
    Kidhome = st.selectbox('Kid in home(1 for yes 0 for no)',('1','0'))
    Teenhome= st.selectbox('Teen in home(1 for yes 0 for no)',('1','0'))
    Cust_Age = st.text_input("Insert the age")
    Num_Accepted_Cmp = st.selectbox('Number of accepted campaign(0 to 5)',('0','1','2','3','4','5'))
    education = st.selectbox('Eduacational qualification(0 for graduate 1 for postgraguate 2 for undergraduate)',('0','1','2'))
    marry= st.selectbox('Married_status(1 for partner 0 for alone)',('1','0'))
    result=""
    if st.button("Submit"):
        result=predict_model(Income, Kidhome, Teenhome, Cust_Age, Num_Accepted_Cmp, education, marry)
    st.success("The output is {}".format(result))

if __name__=='__main__':
    main()
    
