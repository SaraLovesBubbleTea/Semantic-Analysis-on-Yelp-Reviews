# ANLY512_FinalProject
This is a group project about Yelp review semantic analysis. Group members are Xiner Ning and Jin Young Yang. 

The original dataset contains 10,000 restaurant reviews on Yelp. Three different kinds of predictive models (Linear Regression, Logistic Regression, SVM Regression) are used in predicting the star scores. Structure features and semantic features are extracted from the review texts. This project also uses review_style, whcih is included in the original data, to train the models.

Data is from [Yelp](https://www.dropbox.com/s/wc6rzl1a2os721d/yelp.csv?dl=0). A copy of the dataset is also included in the repository. 

## Python code: final_code.py
This file cleans the Yelp dataset, create feature matrix and build predictive models. 
Outputs of this file are models as well as their scores and confusion matrix.  

## LIWC2015_Yelp.csv
This file is generated from Linguistic Inquiry and Word Count (LIWC) analyzer. The LIWC lexicon is proprietary, so it is not included in this repository. More details about how this project integrates the tool is discussed in 'requirements.txt'. The lexicon data can be acquired (purchased) from [liwc.net](http://liwc.wpengine.com/).
