# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:14:27 2019

@author: shera
"""

import argparse
import pandas as pd
import nltk
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

def load_liwcResults(myliwc):
    liwc = pd.read_csv(myliwc)
    return liwc



def load_yelp(myyelp):
    yelp = pd.read_csv(myyelp)
    return yelp
    
def linearModel_baseline(feature_df):
    #feature_df = feature_matrix()
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    
    # Create linear regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']], yelp_train[['stars']])

# Make predictions using the testing set
    reg_score = regr.score(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']], yelp_test[['stars']])
    print ('Score for linearn regression model using baseline features is ',reg_score)

def linearModel_LIWC(feature_df):
    #feature_df = feature_matrix()
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    
    # Create linear regression object
    regr = LinearRegression()
    # Train the model using the training sets
    regr.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']], yelp_train[['stars']])

# Make predictions using the testing set
    reg_score = regr.score(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']], yelp_test[['stars']])
    print ('Score for linearn regression model using LIWC features is ',reg_score)
    
def logisticModel_baseline(feature_df):
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    logisticRegr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #logisticRegr = LogisticRegression()
    logisticRegr.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']], yelp_train[['stars_binary']])
    #regr.score(yelp_test[['reviewLen','SenNum','ExNum','posemo','negemo']], yelp_test.iloc[0:,4])

    y_pred = logisticRegr.predict(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']])
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(yelp_test[['stars_binary']], y_pred)
    print('Confusion matrix for logistic regression model using baseline features is\n ',confusion_matrix)
    
def logisticModel_LIWC(feature_df):
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    logisticRegr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    #logisticRegr = LogisticRegression()
    logisticRegr.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']], yelp_train[['stars_binary']])
    #regr.score(yelp_test[['reviewLen','SenNum','ExNum','posemo','negemo']], yelp_test.iloc[0:,4])

    y_pred = logisticRegr.predict(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']])
    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(yelp_test[['stars_binary']], y_pred)
    print('Confusion matrix for logistic regression model using LIWC features is\n ',confusion_matrix)
    
def SVM_baseline(feature_df):
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    
    # Create linear regression object
    clf = SVR(C=1.0, epsilon=0.2)
    # Train the model using the training sets
    clf.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']], yelp_train[['stars']])

# Make predictions using the testing set
    reg_score = clf.score(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary']], yelp_test[['stars']])
    print ('Score for SVM regression model using baseline features is ',reg_score)

def SVM_LIWC(feature_df):
    yelp_train, yelp_test = train_test_split(feature_df, test_size=0.2,random_state=42)
    
    # Create linear regression object
    clf = SVR(C=1.0, epsilon=0.2)
    # Train the model using the training sets
    clf.fit(yelp_train[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']], yelp_train[['stars']])

# Make predictions using the testing set
    reg_score = clf.score(yelp_test[['reviewLen','SenNum','ExNum','cool_binary','useful_binary','funny_binary','posemo','negemo']], yelp_test[['stars']])
    print ('Score for SVM regression model using LIWC features is ',reg_score)



def main(sts_yelp_file, sts_LIWC_yelpReview_file):
    liwc = load_liwcResults(sts_LIWC_yelpReview_file)
    yelp = load_yelp(sts_yelp_file)
    yelp['posemo'] = liwc['posemo'].tolist()
    yelp['negemo'] = liwc['negemo'].tolist()
    yelp = yelp[(yelp['cool']<10) & (yelp['useful']<10) & (yelp['funny']<10)]    
    reviews = yelp['text'].tolist()
    stars = yelp['stars'].tolist()
    cool = yelp['cool'].tolist()
    useful = yelp['useful'].tolist()
    funny=yelp['funny'].tolist()
    stars_binary =np.asarray(stars) > 3
    cool_binary = np.asarray(cool)>3
    useful_binary=np.asarray(useful)>3
    funny_binary=np.asarray(funny)>2
    
    
    review_length=[] # in terms of token, including , .
    for review in reviews:
        mylength=len(nltk.word_tokenize(review))
        review_length.append(mylength)
    
    #  number of sentences (sentNum)
    sentNum=[]
    for review in reviews:
        mylength=len(nltk.sent_tokenize(review))
        sentNum.append(mylength)

    # percentage of '!' occurs in each review
    ex_occur=[]
    for review in reviews:
        mytoken=nltk.word_tokenize(review)
        num_ex = mytoken.count('!')
        ex_occur.append(num_ex)
        
    
    feature_df = pd.DataFrame(list(zip(review_length,sentNum,ex_occur,stars,stars_binary,cool_binary,useful_binary,funny_binary,yelp['posemo'].tolist(),yelp['negemo'].tolist())), 
                              columns =['reviewLen','SenNum','ExNum','stars','stars_binary','cool_binary','useful_binary','funny_binary','posemo','negemo']) 
    

    linearModel_baseline(feature_df)
    linearModel_LIWC(feature_df)
    logisticModel_baseline(feature_df)
    logisticModel_LIWC(feature_df)
    SVM_baseline(feature_df)
    SVM_LIWC(feature_df)










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_yelp_file", type=str, default="yelp.csv",
                        help="yelp review file")
    parser.add_argument("--sts_LIWC_yelpReview_file", type=str, default="LIWC2015_Yelp.csv",
                        help="LIWC results")
    args = parser.parse_args()

    main(args.sts_yelp_file, args.sts_LIWC_yelpReview_file)
