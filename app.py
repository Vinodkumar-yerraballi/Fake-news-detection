import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data=pd.read_csv('news_articles.csv')
data=data.dropna(how='any')

x=np.array(data['text_without_stopwords'])
y=np.array(data['label'])
cv=CountVectorizer()
x=cv.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=MultinomialNB()
model.fit(x_train,y_train)

import streamlit as st
st.title("Fake news detection system")
def fakenewsdetection():
    user=st.text_area("Enter any news headline :")
    st.button("Predict")
    if len(user)<1:
        st.write(" ")
    else:
        sample = user 
        data=cv.transform([sample]).toarray()
        a=model.predict(data)
        st.title(a)
if __name__=='__main__':
    fakenewsdetection()
