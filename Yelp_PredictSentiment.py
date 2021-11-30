#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tqdm import tqdm
import copy
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nltk


import spacy
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from keras.preprocessing.sequence import pad_sequences

from sklearn.ensemble import RandomForestClassifier
import re
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import operator
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score, recall_score, confusion_matrix
from scipy.sparse import hstack


# In[2]:


#read data
yelp_business = pd.read_json('yelp_academic_dataset_business.json', lines = True)
yelp_review = pd.read_json('yelp_academic_dataset_review_reduced.json', lines = True)


# In[3]:


#filter reviews of restaurants only

def reviews_restaurants(business_data,review_data):
    restaurant_data =  business_data[business_data['categories'].str.contains('Restaurant') == True]
    restaurant_reviews = review_data[review_data.business_id.isin(restaurant_data['business_id']) == True]
    return(restaurant_reviews)


# In[4]:


review_restaurant_data = reviews_restaurants(yelp_business, yelp_review)


# In[5]:


#saving and reading data in csv
review_restaurant_data.to_csv("review_rest_data.csv",index = False)

review_restaurant_data = pd.read_csv("review_rest_data.csv")


# In[6]:


#only read two colums - review and the star rating
review_restaurant_data = review_restaurant_data[["text","stars"]]
review_restaurant_data = review_restaurant_data.reset_index(drop = True)


# In[7]:


## We label the data -If the stars achieved by a restaurant is above 3 then its "Positive" review or else it is a "Negative" review

def label_data(data):
    target = {"Target_sentiment":[]}
    for i in data["stars"]:
        j = 0
        if i > 3:
            j = 1
            target["Target_sentiment"].append(j)
        else:
            target["Target_sentiment"].append(j) 
    data = data.join(pd.DataFrame(target, index = data.index))
    
    #we drop the stars column because it is not useful to us now
    data = data.drop('stars', axis = 1)
    data_sample = data.sample(n = 10000, random_state = 42)
    data_sample = data_sample.reset_index(drop = True)
    return(data_sample)


# In[8]:


restaurant_reviews_after_labels = label_data(review_restaurant_data)


# In[9]:


## Function for replacing contractions with normal words
def replace_contractions(data):
    data = re.sub(r"ain't", "am not", data)
    data = re.sub(r"aren't", "are not", data)
    data = re.sub(r"can't", "can not", data)
    data = re.sub(r"can't've", "can not have", data)
    data = re.sub(r"'cause", "because", data)
    data = re.sub(r"could've", "could have", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub(r"couldn't've", "could not have", data)
    data = re.sub(r"doesn't", "does not", data)
    data = re.sub(r"hadn't", "had not", data)
    data = re.sub(r"hadn't've", "had not have", data)
    data = re.sub(r"hasn't", "has not", data)
    data = re.sub(r"haven't", "have not", data)
    data = re.sub(r"he'd", "he had", data)
    data = re.sub(r"he'd've", "he would have", data)
    data = re.sub(r"he'll", "he will", data)
    data = re.sub(r"he'll've", "he will have", data)
    data = re.sub(r"he's", "he has", data)
    data = re.sub(r"how'd", "how did", data)
    data = re.sub(r"how'd'y", "how do you", data)
    data = re.sub(r"how'll", "how will", data)
    data = re.sub(r"how's", "how has", data)
    data = re.sub(r"i'd", "i had", data)
    data = re.sub(r"i'd've", "i would have", data)
    data = re.sub(r"i'll", "i shall", data)
    data = re.sub(r"i'll've", "i shall have", data)
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"i've", "i have", data)
    data = re.sub(r"isn't", "is not", data)
    data = re.sub(r"it'd", "it had", data)
    data = re.sub(r"it'd've", "it would have", data)
    data = re.sub(r"it'll", "it shall", data)
    data = re.sub(r"it'll've", "it shall have", data)
    data = re.sub(r"it's", "it has", data)
    data = re.sub(r"let's", "let us", data)
    data = re.sub(r"ma'am", "madam", data)
    data = re.sub(r"mayn't", "may not", data)
    data = re.sub(r"might've", "might have", data)
    data = re.sub(r"mightn't", "might not", data)
    data = re.sub(r"mightn't've", "might not have", data)
    data = re.sub(r"must've", "must have", data)
    data = re.sub(r"mustn't", "must not", data)
    data = re.sub(r"mustn't've", "must not have", data)
    data = re.sub(r"needn't", "need not", data)
    data = re.sub(r"needn't've", "need not have", data)
    data = re.sub(r"o'clock", "of the clock", data)
    data = re.sub(r"oughtn't", "ought not", data)
    data = re.sub(r"oughtn't've", "ought not have", data)
    data = re.sub(r"shan't", "shall not", data)
    data = re.sub(r"sha'n't", "shall not", data)
    data = re.sub(r"shan't've", "shall not have", data)
    data = re.sub(r"she'd", "she had", data)
    data = re.sub(r"she'd've", "she would have", data)
    data = re.sub(r"she'll", "she shall", data)
    data = re.sub(r"she'll've", "she shall have", data)
    data = re.sub(r"she's", "she has", data)
    data = re.sub(r"should've", "should have", data)
    data = re.sub(r"shouldn't", "should not", data)
    data = re.sub(r"shouldn't've", "should not have", data)
    data = re.sub(r"so've", "so have", data)
    data = re.sub(r"so's", "so as", data)
    data = re.sub(r"that'd", "that would", data)
    data = re.sub(r"that'd've", "that would have", data)
    data = re.sub(r"that's", "that has", data)
    data = re.sub(r"there'd", "there had", data)
    data = re.sub(r"there'd've", "there would have", data)
    data = re.sub(r"there's", "there has", data)
    data = re.sub(r"they'd", "they had", data)
    data = re.sub(r"they'd've", "they would have", data)
    data = re.sub(r"they'll", "they shall", data)
    data = re.sub(r"they'll've", "they shall have", data)
    data = re.sub(r"they're", "they are", data)
    data = re.sub(r"they've", "they have", data)
    data = re.sub(r"to've", "to have", data)
    data = re.sub(r"wasn't", "was not", data)
    data = re.sub(r"we'd", "we had", data)
    data = re.sub(r"we'd've", "we would have", data)
    data = re.sub(r"we'll", "we will", data)
    data = re.sub(r"we'll've", "we will have", data)
    data = re.sub(r"we're", "we are", data)
    data = re.sub(r"we've", "we have", data)
    data = re.sub(r"weren't", "were not", data)
    data = re.sub(r"what'll", "what shall", data)
    data = re.sub(r"what'll've", "what shall have", data)
    data = re.sub(r"what're", "what are", data)
    data = re.sub(r"what's", "what has", data)
    data = re.sub(r"what've", "what have", data)
    data = re.sub(r"when's", "when has", data)
    data = re.sub(r"when've", "when have", data)
    data = re.sub(r"where'd", "where did", data)
    data = re.sub(r"where's", "where has", data)
    data = re.sub(r"where've", "where have", data)
    data = re.sub(r"who'll", "who shall", data)
    data = re.sub(r"who'll've", "who shall have", data)
    data = re.sub(r"who's", "who has", data)
    data = re.sub(r"who've", "who have", data)
    data = re.sub(r"why's", "why has", data)
    data = re.sub(r"why've", "why have", data)
    data = re.sub(r"will've", "will have", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"won't've", "will not have", data)
    data = re.sub(r"would've", "would have", data)
    data = re.sub(r"wouldn't", "would not", data)
    data = re.sub(r"wouldn't've", "would not have", data)
    data = re.sub(r"y'all", "you all", data)
    data = re.sub(r"y'all'd", "you all would", data)
    data = re.sub(r"y'all'd've", "you all would have", data)
    data = re.sub(r"y'all're", "you all are", data)
    data = re.sub(r"y'all've", "you all have", data)
    data = re.sub(r"you'd", "you had", data)
    data = re.sub(r"you'd've", "you would have", data)
    data = re.sub(r"you'll", "you shall", data)
    data = re.sub(r"you'll've", "you shall have", data)
    data = re.sub(r"how's", "how has", data)
    data = re.sub(r"you're", "you are", data)
    data = re.sub(r"you've", "you have", data)
    data = re.sub(r"didn't", "did not", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"'","",data)
    data = re.sub(r". . .","",data)
    return(data)


# In[10]:


def remove_unnecessary(data):
    for index, row in tqdm(data.iterrows()):
        cleaned_text = ""
        preprocess_word = re.sub(r'([\d]+[a-zA-Z]+)|([a-zA-Z]+[\d]+)', "", row["text"])
        preprocess_word = re.sub(r"(^|\s)(\-?\d+(?:\.\d)*|\d+|[\d]+[A-Za-z]+)"," ", preprocess_word.lower())
        preprocess_word = re.sub('[^A-Za-z\']+', " ", preprocess_word)
        cleaned_text = cleaned_text + preprocess_word
        cleaned_text = replace_contractions(cleaned_text)
        data["text"][index] = cleaned_text
    return(data)


# In[11]:


def remove_stopwords_and_lemmatize(data):
    copy_data = copy.deepcopy(data)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - set(['no', 'not'])
    for index, row in tqdm(copy_data.iterrows()):
        sent = ''
        for e in row["text"].split():
            if e not in stop_words:
                e = lemmatizer.lemmatize(e, pos ="a")
                sent = ' '.join([sent,e])
        copy_data["text"][index] = sent
    return(copy_data)
                 


# In[12]:


restaurants_reviews_preprocessed = remove_unnecessary(restaurant_reviews_after_labels)


# In[13]:


restaurants_reviews_preprocessed_lemmatize = remove_stopwords_and_lemmatize(restaurants_reviews_preprocessed)


# In[14]:



def restructure_data(pre_processed_data):
    cleaned_data = copy.deepcopy(pre_processed_data)
    cleaned_data["sentiment_polarity"] = cleaned_data["text"].map(lambda text: TextBlob(text).sentiment.polarity)
    cleaned_data["text_length"] = cleaned_data["text"].astype(str).apply(len)
    cleaned_data["Word_count"] = cleaned_data["text"].apply(lambda x: len(str(x).split()))    
    return(cleaned_data)
cleaned_data = restructure_data(restaurants_reviews_preprocessed_lemmatize)


# In[17]:


## This code is for spliting the data for train, dev and test set
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_data.drop(columns = ["Target_sentiment","sentiment_polarity","Word_count"], axis = 1), restaurants_reviews_preprocessed_lemmatize["Target_sentiment"],
                                                   test_size = 0.2, random_state = 40, stratify = restaurants_reviews_preprocessed_lemmatize["Target_sentiment"] )

X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train,
                                                   test_size = 0.2, random_state = 40, stratify = Y_train)


# In[19]:


#remove this, this is not required
print("Shape of the train set:", X_train.shape, Y_train.shape)
print("Shape of the dev set:", X_dev.shape, Y_dev.shape)
print("shape of the test set:", X_test.shape, Y_test.shape )


# In[20]:


## Function for creating tf-idf vectors from text

def tfidf(train,dev,test):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2),max_features = 3000)
    tfidf_vectorizer.fit(train['text'].values)
    
    X_train_reviews_tfidf = tfidf_vectorizer.transform(train['text'].values)
    X_dev_reviews_tfidf = tfidf_vectorizer.transform(dev['text'].values)
    X_test_reviews_tfidf = tfidf_vectorizer.transform(test['text'].values)
    
    print("After vectorizations")
    print(X_train_reviews_tfidf.shape)
    print(X_dev_reviews_tfidf.shape)
    print(X_test_reviews_tfidf.shape)
    print("="*100)
    return(X_train_reviews_tfidf,X_dev_reviews_tfidf,X_test_reviews_tfidf )


# In[21]:


## Function for performing standarization on the numerical feature

def numerical_feature_standardization(train,dev,test):
    normaliser = Normalizer()
    normaliser.fit(train['text_length'].values.reshape(-1,1))
    
    
    X_train_text_len_stand = normaliser.transform(train['text_length'].values.reshape(-1,1))
    X_dev_text_len_stand = normaliser.transform(dev['text_length'].values.reshape(-1,1))
    X_test_text_len_stand = normaliser.transform(test['text_length'].values.reshape(-1,1))
    print("After Normalization")
    print(X_train_text_len_stand.shape)
    print(X_dev_text_len_stand.shape)
    print(X_test_text_len_stand.shape)
    print("="*100)
    
    return(X_train_text_len_stand,X_dev_text_len_stand, X_test_text_len_stand)


# In[22]:


## Calling the above tfidf_vec function to create 3000 dimensional bigram feature
X_train_reviews_tfidf,X_dev_text_len_stand_tfidf,X_test_reviews_tfidf = tfidf(X_train,X_dev,X_test)


# In[23]:


## calling the above function to normalise the numerical feature text length
X_train_text_len_stand,X_dev_text_len_stand, X_test_text_len_stand = numerical_feature_standardization(X_train,X_dev,X_test)


# In[24]:


## Fuction for merging textual vectors and numerical feature
def merge_text_vectors_and_numerical_features(train1,train2,dev1,dev2,test1,test2,tx):
    train_datam = hstack((train1,train2)).tocsr()
    dev_datam = hstack((dev1,dev2)).tocsr()
    test_datam = hstack((test1,test2 )).tocsr()
    
    
    print(tx +"final data matrix developed")
    print(train_datam.shape)
    print(dev_datam.shape)
    print(test_datam.shape)
    print("="*100)
    
    return(train_datam,dev_datam,test_datam)


# In[25]:


###### Calling the above function to merge tfidf vectors and numerical text length feature
train_data_tfidf, dev_data_tfidf, test_data_tfidf = merge_text_vectors_and_numerical_features(X_train_reviews_tfidf,X_train_text_len_stand,X_dev_text_len_stand_tfidf,X_dev_text_len_stand,X_test_reviews_tfidf, X_test_text_len_stand,tx = "TFIDF ",)


# In[26]:


#### Function for evaluating models

def evaluate_models(y_test,y_pred):
    confusion_matrix_given = confusion_matrix(y_test,y_test_pred)
    sns.heatmap(confusion_matrix_given, annot = True, fmt = 'd',cmap="Blues")
    plt.title('Confusion matrix for Test data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    print("Precision Score of the model:", precision_score(y_test,y_pred)*100)
    print("Recall Score of the model:", recall_score(y_test,y_pred)*100)
    print("Accuracy score of the model:",accuracy_score(y_test,y_pred)*100)
    print("F1 score of the model:",f1_score(y_test,y_pred)*100)
    
    
        


# In[39]:


## Initilizaing multinomial naive bayes model and fitting the train data
nb_model =  MultinomialNB()
nb_model = nb_model.fit(train_data_tfidf, Y_train)


# In[40]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = nb_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred )


# In[41]:


## Initilizaing decision tree classifier and fitting the train data
dc_model = DecisionTreeClassifier()
dc_model = dc_model.fit(train_data_tfidf,Y_train)


# In[42]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = dc_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred )


# In[43]:


## Initilizaing support vector classifier and fitting the train data
svc_model = svm.SVC()
svc_model = svc_model.fit(train_data_tfidf,Y_train)


# In[44]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = svc_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred )


# In[45]:


## Initilizaing random forest classifier and fitting the train data
rf_model =  RandomForestClassifier()
rf_model = rf_model.fit(train_data_tfidf, Y_train)


# In[47]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = rf_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred )


# In[48]:


## function for oversampling the train dataset
def oversampling_data(train_s,y_trains):
    random_oversampler = RandomOverSampler(random_state=0)
    train_data1, y_train1 = random_oversampler.fit_resample(train_s, y_trains)
    return(train_data1, y_train1)
    


# In[50]:


## Calling the above function to oversample the train dataset with tfidf and text length feature
train_data1_tfidf,y_train1_tfidf = oversampling_data(train_data_tfidf,Y_train)


# In[51]:


## Function for hypetuning the parameters for the algorithm by using gridsearch cross validation
def hyper_tuning_grid_search_cross_validation(t_d,y_t,alpha,parameters):
    clf = GridSearchCV(alpha, param_grid= parameters, cv=5, scoring='f1',return_train_score= True)
    hyper = clf.fit(t_d,y_t)
    print("Best parameters for the algorithm", hyper.best_estimator_)
    print("Best cross validation score :", hyper.best_score_)
    return(hyper.best_estimator_)


# In[52]:


## Calling the above function for tuning multinomial naive bayes algorithm using gridsearch cv then fitting the train dataset with best parameter
nb_model =  hyper_tuning_grid_search_cross_validation(train_data1_tfidf,y_train1_tfidf,alpha = MultinomialNB(fit_prior=True, class_prior=None),parameters = {'alpha':[1000,500,100,50,10,5,0.5,1, 0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]})
nb_model = nb_model.fit(train_data1_tfidf, y_train1_tfidf)


# In[54]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = nb_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred)


# In[55]:


## Calling the above function for tuning decision tree classifier using gridsearch cv then fitting the train dataset with best parameter
dt_model =  hyper_tuning_grid_search_cross_validation(train_data1_tfidf,y_train1_tfidf,alpha = DecisionTreeClassifier(),parameters = {"max_features":[1,2,3,4,5],"max_depth":[int(x) for x in range(10)],"min_samples_leaf":[1,2,3,4,5],"min_samples_split":[1,2,3,4,5],"criterion":["gini","entropy"]})
dt_model = dt_model.fit(train_data1_tfidf, y_train1_tfidf)


# In[56]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = dt_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred)


# In[57]:


## Calling the above function for tuning Support vector classifier algorithm using gridsearch cv then fitting the train dataset with best parameter
svc_model = hyper_tuning_grid_search_cross_validation(train_data1_tfidf,y_train1_tfidf,alpha =  SGDClassifier(loss = 'hinge'),parameters = {'alpha':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],'penalty' :[ 'l2','l1'],'max_iter':[20]})
svc_model = svc_model.fit(train_data1_tfidf, y_train1_tfidf)


# In[58]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = svc_model.predict(test_data_tfidf)
evaluate_models(Y_test,y_test_pred)


# In[ ]:


## Calling the above function for tuning Random Forest classifier algorithm using gridsearch cv then fitting the train dataset with best parameter
rf_model = hyper_tuning_grid_search_cross_validation(train_data1_tfidf,y_train1_tfidf,alpha =  RandomForestClassifier(n_jobs = -1),parameters = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],'max_depth':[int(x) for x in np.linspace(10, 110, num = 11)],'min_samples_split':[2, 5, 10],'min_samples_leaf':[1, 2, 4]})
rf_model = rf_model.fit(train_data1_tfidf, y_train1_tfidf)


# In[ ]:


## prediction on the test dataset and then evalauting the model performance
y_test_pred = model.predict(test_data_tfidf)
evaluate(y_test,y_test_pred)


# In[ ]:




