import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('diabetes.csv')
X = df.drop(columns=['Outcome'])
y =df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


st.title('SVM Kernel Selection for Diabetes Checking Accuracy')


kernel = st.selectbox('Choose Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))


if st.button('Submit'):
    # Train the SVM model with the chosen kernel
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)

  
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

 
    st.write(f'Training Accuracy: {train_accuracy:.2f}')
    st.write(f'Testing Accuracy: {test_accuracy:.2f}')

