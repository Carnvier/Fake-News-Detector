import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


LR = LogisticRegression()
DT = DecisionTreeClassifier()
GB = GradientBoostingClassifier(random_state = 0)
# Initialize the TfidfVectorizer
vectorization = TfidfVectorizer()
RF = RandomForestClassifier(random_state = 0)

def wordopt(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub(r"\W", " ", text)  # Remove non-word characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    return text

def train_model():
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')
    data_fake["class"] = 0
    data_true["class"] = 1

    # Selecting the last 10 rows from data_fake
    data_fake_manual_testing = data_fake.tail(10)

    # Dropping rows from data_fake
    for i in range(23480, 23470, -1):
        data_fake.drop([i], axis=0, inplace=True)

    # Selecting the last 10 rows from data_true
    data_true_manual_testing = data_true.tail(10)

    # Dropping rows from data_true
    for i in range(21416, 21406, -1):
        data_true.drop([i], axis=0, inplace=True)

    data_fake_manual_testing["class"] = 0
    data_true_manual_testing["class"] = 1

    data_merge = pd.concat([data_fake, data_true], axis = 0)

    data = data_merge.drop(['title', 'subject', 'date'], axis = 1)

    data = data.sample(frac = 1)

    data.reset_index(inplace=True)
    data.drop(['index'], axis = 1, inplace = True)

    data['text'] = data['text'].apply(wordopt)

    x = data['text']
    y = data['class']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)





    # Fit and transform the training data
    xv_train = vectorization.fit_transform(x_train)

    # Transform the test data
    xv_test = vectorization.transform(x_test)

    
    LR.fit(xv_train, y_train)

    pred_lr = LR.predict(xv_test)

    LR.score(xv_test, y_test)

    print(classification_report(y_test, pred_lr))

   
    DT.fit(xv_train, y_train)

    pred_dt = DT.predict(xv_test)

    DT.score(xv_test, y_test)

    print(classification_report(y_test, pred_dt))

    
    GB.fit(xv_train, y_train)


    predict_gb = GB.predict(xv_test)

    GB.score(xv_test, y_test)

    print(classification_report(y_test, predict_gb))

    

    
    RF.fit(xv_train, y_train)

    pred_rf = RF.predict(xv_test)

    RF.score(xv_test, y_test)

    print(classification_report(y_test, pred_rf))


train_model()