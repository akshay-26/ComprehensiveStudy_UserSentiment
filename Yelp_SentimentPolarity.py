#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
import requests, re
import pandas as pd
import seaborn as sns
import nltk
import string, itertools
from collections import Counter, defaultdict
from nltk.text import Text
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from sklearn.cluster import KMeans
from wordcloud import WordCloud

pd.set_option('display.max_columns',None)
pd.options.display.max_seq_items = 2000
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[6]:


# ### Clean Yelp_business dataset 

# In[2]:


restaurantData = pd.read_csv('yelp_business.csv')

# In[3]:


## drop unuseful column 'neighborhood' 
restaurantData.drop(['neighborhood'], axis=1, inplace=True)

## remove quotation marks in name and address column
restaurantData.name=restaurantData.name.str.replace('"','')
restaurantData.address=restaurantData.address.str.replace('"','')

## filter restaurants of US
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
usa=restaurantData.loc[restaurantData['state'].isin(states)]
restaurantData.head()

## select all restaurants in USA
restaurants=usa[usa['categories'].str.contains('Restaurants')]


# In[7]:


## select out 16 cuisine types of restaurants and rename the category
restaurants.is_copy=False
restaurants['category']=pd.Series()
restaurants.loc[restaurants.categories.str.contains('American'),'category'] = 'American'
restaurants.loc[restaurants.categories.str.contains('Mexican'), 'category'] = 'Mexican'
restaurants.loc[restaurants.categories.str.contains('Italian'), 'category'] = 'Italian'
restaurants.loc[restaurants.categories.str.contains('Japanese'), 'category'] = 'Japanese'
restaurants.loc[restaurants.categories.str.contains('Chinese'), 'category'] = 'Chinese'
restaurants.loc[restaurants.categories.str.contains('Thai'), 'category'] = 'Thai'
restaurants.loc[restaurants.categories.str.contains('Mediterranean'), 'category'] = 'Mediterranean'
restaurants.loc[restaurants.categories.str.contains('French'), 'category'] = 'French'
restaurants.loc[restaurants.categories.str.contains('Vietnamese'), 'category'] = 'Vietnamese'
restaurants.loc[restaurants.categories.str.contains('Greek'),'category'] = 'Greek'
restaurants.loc[restaurants.categories.str.contains('Indian'),'category'] = 'Indian'
restaurants.loc[restaurants.categories.str.contains('Korean'),'category'] = 'Korean'
restaurants.loc[restaurants.categories.str.contains('Hawaiian'),'category'] = 'Hawaiian'
restaurants.loc[restaurants.categories.str.contains('African'),'category'] = 'African'
restaurants.loc[restaurants.categories.str.contains('Spanish'),'category'] = 'Spanish'
restaurants.loc[restaurants.categories.str.contains('Middle_eastern'),'category'] = 'Middle_eastern'
restaurants.category[:20]


# In[8]:


# In[6]:


## drop null values in category, delete original column categories and reset the index
restaurants=restaurants.dropna(axis=0, subset=['category'])
del restaurants['categories']
restaurants=restaurants.reset_index(drop=True)
restaurants.head(10)


# In[7]:


# In[9]:


## check total number of us restaurants
restaurants.shape


# In[8]:


## check whether has duplicated restaurantData id
restaurants.business_id.duplicated().sum()


# In[9]:


## check the datatype
restaurants.dtypes


# In[10]:


## check missing values
restaurants.isnull().sum()


# ### Clean yelp_review dataset

# In[11]:


## load feedback details
feedback = pd.read_csv('yelp_review.csv')
feedback.head()


# In[12]:


## check missing values
feedback.isnull().sum()


# In[13]:


## check duplicates of review_id
feedback.review_id.duplicated().sum()


# ### Merge two datasets and get new dataframe resFeedback


# In[10]:


# In[14]:


## merge restaurantData details and feedback details
resFeedback = pd.merge(restaurants, feedback, on = 'business_id')

## update column names
resFeedback.rename(columns={'stars_x':'avg_star','stars_y':'rating'}, inplace=True)

## add column of number of words in feedback and label of negative and postive reviews
resFeedback['num_words_review'] = resFeedback.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))
    


# In[15]:


## add column of number of words in feedback and label of negative and postive reviews
resFeedback['num_words_review'] = resFeedback.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','').map(lambda x: len(x.split()))


# In[16]:


# label reviews as positive or negative
resFeedback['labels'] = ''
resFeedback.loc[resFeedback.rating >=4, 'labels'] = 'positive'
# resFeedback.loc[resFeedback.rating ==3, 'labels'] = 'neural'
resFeedback.loc[resFeedback.rating <=3, 'labels'] = 'negative'

# drop neutral reviews for easy analysis
# resFeedback.drop(resFeedback[resFeedback['labels'] =='neural'].index, axis=0, inplace=True)
# resFeedback.reset_index(drop=True, inplace=True)

resFeedback.head()


# In[11]:


# ## Exploratory Data Analysis

# ### Restaurants Distribution

# #### Distribution of restaurants in each category

# In[17]:


plt.style.use('ggplot')


# In[18]:


plt.figure(figsize=(11,7))
restaurantSet = restaurants.category.value_counts()
sns.countplot(y='category',data=restaurants, 
              order = restaurantSet.index, palette= sns.color_palette("RdBu_r", len(restaurantSet)))
plt.xlabel('Number of restaurants', fontsize=14, labelpad=10)
plt.ylabel('Category', fontsize=14)
plt.title('Count of Restaurants by Category', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(restaurants.category.value_counts()):
    plt.text(v, i+0.15, str(v), fontweight='bold', fontsize=14)


# Categories in dark blue color have the largest number of restaurants. On the contrary, categories in dark red color have the least number of restaurants. The top 5 type of restaurants are American, Mexican, Italian, Chinese and Japanese. 

# #### Top 10 cities with most restaurants

# In[19]:


# In[12]:


plt.figure(figsize=(11,6))
restaurantSet = restaurants.city.value_counts()[:10]
sns.barplot(restaurantSet.index, restaurantSet.values, palette=sns.color_palette("GnBu_r", len(restaurantSet)))
plt.ylabel('Number of restaurants', fontsize=14, labelpad=10)
plt.xlabel('City', fontsize=14, labelpad=10)
plt.title('Count of Restaurants by City (Top 10)', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# #### Distribution of restaurants in each state

# In[20]:


plt.figure(figsize=(11,6))
restaurantSet = restaurants.state.value_counts()
sns.barplot(restaurantSet.index, restaurantSet.values,palette=sns.color_palette("GnBu_r", len(restaurantSet)) )
plt.ylabel('Number of restaurants', fontsize=14)
plt.xlabel('State', fontsize=14)
plt.title('Count of Restaurants by State', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center', fontweight='bold', fontsize=14)


# ### Reviews Distribution

# #### Distribution of reviews by cuisine type

# In[21]:


plt.figure(figsize=(11,7))
restaurantSet = restaurants.groupby('category')['review_count'].sum().sort_values(ascending = False)
sns.barplot(y=restaurantSet.index, x= restaurantSet.values, palette= sns.color_palette("RdBu_r", len(restaurantSet)) )
plt.ylabel('Category', fontsize=14)
plt.xlabel('Count of reviews', fontsize=14)
plt.title('Count of Reviews by Cuisine Type', fontsize=15)
for i,v in enumerate(restaurantSet):
    plt.text(v, i+0.15, str(v),fontweight='bold', fontsize=14)
plt.tick_params(labelsize=14)


# #### Top 10 cities with most reviews

# In[22]:


# In[13]:


plt.figure(figsize=(11,6))
restaurantSet = restaurants.groupby('city')['review_count'].sum().sort_values(ascending=False)[:10]
sns.barplot(restaurantSet.index, restaurantSet.values, palette=sns.color_palette("GnBu_r", len(restaurantSet)) )
plt.xlabel('City', labelpad=10, fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Count of Reviews by City (Top 10)', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# In[14]:


# #### Top 9 restaurants with most reviews

# In[23]:


plt.figure(figsize=(11,6))
restaurantSet = restaurants[['name','review_count']].sort_values(by='review_count', ascending=False)[:9]
sns.barplot(x=restaurantSet.review_count, y = restaurantSet.name, palette=sns.color_palette("GnBu_r", len(restaurantSet)), ci=None)
plt.xlabel('Count of Review', labelpad=10, fontsize=14)
plt.ylabel('Restaurants', fontsize=14)
plt.title('TOP 9 Restaurants with Most Reviews', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)
for  i, v in enumerate(restaurantSet.review_count):
    plt.text(v, i, str(v), fontweight='bold', fontsize=14)


# #### Distribution of positive and negative reviews in each category

# In[24]:


# In[15]:


details = pd.pivot_table(resFeedback, values=["review_id"], index=["category"],columns=["labels"], 
                       aggfunc=len, margins=True, dropna=True,fill_value=0)
tPercent = details.div( details.iloc[:,-1], axis=0).iloc[:-1,-2].sort_values(ascending=False)


# In[25]:


# In[16]:


details = pd.pivot_table(resFeedback, values=["review_id"], index=["category"],columns=["labels"], 
                       aggfunc=len, margins=True, dropna=True,fill_value=0)
tPercent = details.div( details.iloc[:,-1], axis=0).iloc[:-1,-2].sort_values(ascending=False)
plt.figure(figsize=(11,8))
plt.subplot(211)
sns.pointplot(x=tPercent.index, y= tPercent.values)
plt.xlabel('Category', labelpad=7, fontsize=14)
plt.ylabel('Percentage of positive reviews', fontsize=14)
plt.title('Percentage of Positive Reviews', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=40)
for  i, v in enumerate(tPercent.round(2)):
    plt.text(i, v*1.001, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)
    


# In[17]:


plt.subplot(212)
restaurantSet = resFeedback.groupby('category')['rating'].mean().round(2).sort_values(ascending=False)
sns.pointplot(restaurantSet.index, restaurantSet.values)
plt.ylim(3)
plt.xlabel('Catagory', labelpad=10, fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.title('Average Rating of each Category', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=40)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)
    
plt.subplots_adjust(hspace=1)


# #### Average length of reviews

# #### Average length of words in each category

# In[26]:


# In[18]:


details = resFeedback.groupby(['category','labels'])['num_words_review'].mean().round().unstack()
plt.figure(figsize=(11,8))
sns.heatmap(details, cmap='YlGnBu', fmt='g',annot=True, linewidths=1)
plt.tick_params(labelsize=15)


# ### Ratings Distribution

# #### Distribution of ratings by restaurants

# In[27]:


# In[19]:


plt.figure(figsize=(11,6))
restaurantSet = restaurants.stars.value_counts().sort_index()
sns.barplot(restaurantSet.index, restaurantSet.values, palette=sns.color_palette("RdBu_r", len(restaurantSet)))
plt.xlabel('Average Rating', labelpad=10, fontsize=14)
plt.ylabel('Count of restaurants', fontsize=14)
plt.title('Count of Restaurants against Ratings', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# #### Distribution of ratings by reviews

# In[28]:


# In[20]:


plt.figure(figsize=(11,7))
restaurantSet = resFeedback.rating.value_counts().sort_index()
sns.barplot(restaurantSet.index, restaurantSet.values, palette=sns.color_palette("RdBu_r", len(restaurantSet)))
plt.xlabel('Review Rating', labelpad=10, fontsize=14)
plt.ylabel('Count of reviews', fontsize=14)
plt.title('Count of Reviews against Rating', fontsize=15)
plt.tick_params(labelsize=14)
for  i, v in enumerate(restaurantSet):
    plt.text(i, v*1.02, str(v), horizontalalignment ='center',fontweight='bold', fontsize=14)


# ## Review Analysis

# ### Positive words and negative words

# In[29]:


# In[21]:


import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[30]:


## convert text to lower case
resFeedback.text = resFeedback.text.str.lower()

## remove unnecessary punctuation
resFeedback['removed_punct_text']= resFeedback.text.str.replace('\n','').                                           str.replace('[!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~]','')


# In[31]:


# In[22]:


## import positive file which contains common meaningless positive words such as good
fLikeWords = open('positive.txt')
fileP =csv.reader(fLikeWords)
likes = [word[0] for word in fileP]

## import negative file which contains common meaningless positive words such as bad
fDislikeWords = open('negative.txt')
fileP =csv.reader(fDislikeWords)
dislikes = [word[0] for word in fileP]


# In[32]:


# In[23]:


## get dataset by category
def fetchDetails(types):
    dataframe = resFeedback[['removed_punct_text','labels']][resFeedback.category==types]
    dataframe.reset_index(drop=True, inplace =True)
    dataframe.rename(columns={'removed_punct_text':'text'}, inplace=True)
    return dataframe



## only keep positive and negative words
def remove(feedback):
    words = [word for word in feedback.split() if word in likes + dislikes]
    words = ' '.join(words)
    return words


# In[24]:


Korean_reviews = fetchDetails('Korean')
Korean_train, Korean_test = train_test_split(Korean_reviews[['text','labels']],test_size=0.5)
print('Total %d number of reviews' % Korean_train.shape[0])


# In[25]:


def divide(dataset, test_size):
    df_train, df_test = train_test_split(dataset[['text','labels']],test_size=test_size)
    return df_train


# In[28]:


## filter words
Korean_train.text = Korean_train.text.apply(remove)
## construct features and labels
class_train=list(Korean_train['text'])
TrainingCategory=list(Korean_train['labels'])

terms_test=list(Korean_test['text'])
class_test=list(Korean_test['labels'])

## get bag of words : the frequencies of various words appeared in each review
vectorizer = CountVectorizer()
feature_train_counts=vectorizer.fit_transform(class_train)
feature_train_counts.shape


# In[43]:


## run model
supportVectorM = LinearSVC()
supportVectorM.fit(feature_train_counts, TrainingCategory)


# Support Vector Machine (SVM) model was applied to differentiate positive and
# negative words in reviews, and further to get a word score to understand how positive or how negative the words are.

# ### Now we can calculate polarity score of each word in the specific category

# #### Korean

# In[76]:
## create dataframe for score of each word in a review calculated by svm model
coeff = supportVectorM.coef_[0]
print('coeff', coeff)
Korean_words_score = pd.DataFrame({'score': coeff, 'word': vectorizer.get_feature_names()})
print("Korean_words_score: \n",Korean_words_score)

## get frequency of each word in all

Korean_reviews = pd.DataFrame(feature_train_counts.toarray(), columns=vectorizer.get_feature_names())
Korean_reviews['labels'] = class_train
Korean_frequency = Korean_reviews[Korean_reviews['labels'] =='positive'].sum()[:-1]
print("Korean_frequency: \n", Korean_frequency)
Korean_words_score.set_index('word', inplace=True)
Korean_polarity_score = Korean_words_score
Korean_polarity_score['frequency'] = Korean_frequency
print("Korean_polarity_score: \n",Korean_polarity_score)

## calculate polarity score 
Korean_polarity_score['polarity'] = Korean_polarity_score.score * Korean_polarity_score.frequency / Korean_reviews.shape[0]
print("Korean_polarity_score: \n",Korean_polarity_score[:100])


# In[30]:


## drop unnecessary words
ineffective_positive_words = Korean_polarity_score.loc[['great','amazing','love','best','awesome','excellent','good',
                                                    'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous']]
ineffective_negative_words =  Korean_polarity_score.loc[['bad','disappointed','unfortunately','disappointing','horrible',
                                                     'lacking','terrible','sorry', 'disappoint']]
Korean_polarity_score.drop(ineffective_positive_words.index, axis=0, inplace=True)
Korean_polarity_score.drop(ineffective_negative_words.index, axis=0, inplace=True)
# In[84]:

Korean_polarity_score.polarity = Korean_polarity_score.polarity.astype(float)
Korean_polarity_score.frequency = Korean_polarity_score.frequency.astype(float)

# In[85]:

Korean_polarity_score[Korean_polarity_score.polarity>0].sort_values('polarity', ascending=False)[:20]


# #### Get top 10 most informative positive and negative words

# In[51]:


Korean_top_positive_words = ['delicious','friendly','attentive','recommend','fresh','variety','reasonable','tender','clean','authentic']
Korean_top_negative_words = ['bland','slow','expensive','overpriced', 'cold', 'greasy','sweet','fatty','rude','dirty']
Korean_top_review_words = Korean_polarity_score.loc[Korean_top_positive_words+Korean_top_negative_words,'polarity']


# In[52]:


# In[44]:


plt.figure(figsize=(11,6))
colors = ['red' if c < 0 else 'blue' for c in Korean_top_review_words.values]
sns.barplot(y=Korean_top_review_words.index, x=Korean_top_review_words.values, palette=colors)
plt.xlabel('Polarity Score', labelpad=10, fontsize=14)
plt.ylabel('Words', fontsize=14)
plt.title('TOP 10 Positive and Negative Words in Korean Restaurants', fontsize=15)
plt.tick_params(labelsize=14)
plt.xticks(rotation=15)


# In[53]:


# In[50]:


def retrieve_polarity_score(dataset):
    dataset.text = dataset.text.apply(remove)
    
    trainingWords=list(dataset['text'])
    TrainingCategory=list(dataset['labels'])
    
    ## get bag of words
    v = CountVectorizer()
    freqTrain=v.fit_transform(trainingWords)
    
    ## run model
    supportVectorM = LinearSVC()
    supportVectorM.fit(freqTrain, TrainingCategory)
    
    ## create dataframe for score of each word in a feedback calculated by supportVectorM model
    coeff = supportVectorM.coef_[0]
    cuisine_words_score = pd.DataFrame({'score': coeff, 'word': v.get_feature_names()})
    
    ## get frequency of each word in all reviews in specific category
    cuisine_reviews = pd.DataFrame(freqTrain.toarray(), columns=v.get_feature_names())
    cuisine_reviews['labels'] = TrainingCategory
    cuisine_frequency = cuisine_reviews[cuisine_reviews['labels'] =='positive'].sum()[:-1]
    
    cuisine_words_score.set_index('word', inplace=True)
    cuisine_polarity_score = cuisine_words_score
    cuisine_polarity_score['frequency'] = cuisine_frequency
    
    cuisine_polarity_score.score = cuisine_polarity_score.score.astype(float)
    cuisine_polarity_score.frequency = cuisine_polarity_score.frequency.astype(int)
    
    ## calculate polarity score 
    cuisine_polarity_score['polarity'] = cuisine_polarity_score.score * cuisine_polarity_score.frequency / cuisine_reviews.shape[0]
    
    cuisine_polarity_score.polarity = cuisine_polarity_score.polarity.astype(float)
    ## drop unnecessary words
    ineffective_positive_words = ['great','amazing','love','best','awesome','excellent','good',
                                                   'favorite','loved','perfect','gem','perfectly','wonderful',
                                                    'happy','enjoyed','nice','well','super','like','better','decent','fine',
                                                    'pretty','enough','excited','impressed','ready','fantastic','glad','right',
                                                    'fabulous', 'delicious']
    ineffective_negative_words =  ['bad','disappointed','unfortunately','disappointing','horrible',
                                                    'lacking','terrible','sorry']
    ineffective_words = ineffective_positive_words + ineffective_negative_words
    cuisine_polarity_score.drop(cuisine_polarity_score.loc[ineffective_words].index, axis=0, inplace=True)
    
    return cuisine_polarity_score


# In[54]:


# In[51]:


def plot_top_words(top_words, category):
    plt.figure(figsize=(11,6))
    colors = ['red' if c < 0 else 'blue' for c in top_words.values]
    sns.barplot(y=top_words.index, x=top_words.values, palette=colors)
    plt.xlabel('Polarity Score', labelpad=10, fontsize=14)
    plt.ylabel('Words', fontsize=14)
    plt.title('TOP 10 Positive and Negative Words in %s Restaurants ' % category, fontsize=15)
    plt.tick_params(labelsize=14)
    plt.xticks(rotation=15)


# In[55]:


def get_top_words(dataset, label, number=20):
    if label == 'positive':
        dataframe = dataset[dataset.polarity>0].sort_values('polarity',ascending = False)[:number]
    else:
        dataframe = dataset[dataset.polarity<0].sort_values('polarity')[:number]
    return dataframe


# #### Japanese

# In[56]:


# In[52]:


Japanese_reviews = fetchDetails('Japanese')
Japanese_train = divide(Japanese_reviews, 0.9)
print('Total %d number of reviews' % Japanese_train.shape[0])


# In[57]:


Japanese_Cuisine_polarity_score = retrieve_polarity_score(Japanese_train)


# In[58]:


get_top_words(Japanese_Cuisine_polarity_score, 'positive')


# In[59]:


get_top_words(Japanese_Cuisine_polarity_score,'negative',20)


# In[60]:


Japanese_top_positive_words = ['delicious','friendly','fresh','recommend','fun','reasonable',
                               'creative','clean','variety','attentive']
Japanese_top_negative_words = ['hard','cold','wrong','slow','bland','dark','expensive',
                               'rude','overpriced','crowded']
Japanese_top_review_words = Japanese_Cuisine_polarity_score.loc[Japanese_top_positive_words+Japanese_top_negative_words,'polarity']


# In[61]:


plot_top_words(Japanese_top_review_words,'Japanese')


# In[35]:


# #### Thai

# In[62]:


Thai_reviews = fetchDetails('Thai')
Thai_train = divide(Thai_reviews, 0.8)
print('Total %d number of reviews' % Thai_train.shape[0])


# In[65]:


Thai_Cuisine_polarity_score = retrieve_polarity_score(Thai_train)


# In[66]:


get_top_words(Thai_Cuisine_polarity_score,'positive')


# In[67]:


get_top_words(Thai_Cuisine_polarity_score,'negative')


# In[68]:


Thai_top_positive_words = ['delicious','friendly','fresh','recommend','reasonable','affordable','variety',
                           'attentive','fast','comfortable']
Thai_top_negative_words = ['bland','greasy','expensive','weird','wrong','slow','hard','cold','sour','mushy','mess']
Thai_top_words = Thai_Cuisine_polarity_score.loc[Thai_top_positive_words+Thai_top_negative_words,'polarity']
plot_top_words(Thai_top_words, 'Thai')


# #### Chinese

# In[69]:


# In[36]:


Chinese_reviews = fetchDetails('Chinese')
Chinese_train = divide(Chinese_reviews, 0.85)
print('Total %d number of reviews' % Chinese_train.shape[0])


# In[70]:


Chinese_Cuisine_polarity_score = retrieve_polarity_score(Chinese_train)


# In[71]:


get_top_words(Chinese_Cuisine_polarity_score,'positive')


# In[72]:


get_top_words(Chinese_Cuisine_polarity_score,'negative')


# In[73]:


Chinese_top_positive_words = ['delicious','friendly','fresh','authentic','reasonable','hot','fun',
                           'fast','tender','recommend']
Chinese_top_negative_words = ['sour','bland','cold','greasy','hard','slow','wrong','rude','overpriced','frozen']
Chinese_top_words = Chinese_Cuisine_polarity_score.loc[Chinese_top_positive_words+Chinese_top_negative_words,'polarity']
plot_top_words(Chinese_top_words, 'Chinese')


# In[ ]:





# In[37]:


# #### Vietnamese

# In[74]:


Vietnamese_reviews = fetchDetails('Vietnamese')
Vietnamese_train = divide(Vietnamese_reviews, 0.7)
print('Total %d number of reviews' % Vietnamese_train.shape[0])


# In[75]:


Vietnamese_Cuisine_polarity_score = retrieve_polarity_score(Vietnamese_train)


# In[76]:


get_top_words(Vietnamese_Cuisine_polarity_score,'positive')


# In[77]:


get_top_words(Vietnamese_Cuisine_polarity_score,'negative')


# In[78]:


Viet_top_positive_words = ['delicious','fresh','clean','fast','recommend','reasonable','tender',
                           'fancy','refreshing','generous']
Viet_top_negative_words = ['bland','wrong','hard','slow','expensive','rude','greasy','dirty','weird','smelled']
Viet_top_words = Vietnamese_Cuisine_polarity_score.loc[Viet_top_positive_words+Viet_top_negative_words,'polarity']
plot_top_words(Viet_top_words,'Viet')


# #### French

# In[79]:


# In[38]:


French_reviews = fetchDetails('French')
French_train = divide(French_reviews, 0.7)
print('Total %d number of reviews' % French_train.shape[0])


# In[80]:


French_Cuisine_polarity_score = retrieve_polarity_score(French_train)


# In[81]:


get_top_words(French_Cuisine_polarity_score,'positive')


# In[82]:


get_top_words(French_Cuisine_polarity_score, 'negative')


# In[83]:


French_top_positive_words = ['delicious','sweet','tender','impeccable','recommend','rich','attentive',
                             'beautifully','crisp','romantic']
French_top_negative_words = ['cold','expensive','slow','bland','overpriced','mediocre','wrong',
                             'poor','squash','knife']
French_top_words = French_Cuisine_polarity_score.loc[French_top_positive_words+French_top_negative_words,'polarity']
plot_top_words(French_top_words,'French')


# #### Italian

# In[86]:


# In[39]:


Italian_reviews = fetchDetails('Italian')
Italian_train = divide(Italian_reviews, 0.9)
print('Total %d number of reviews' % Italian_train.shape[0])


# In[87]:


Italian_Cuisine_polarity_score = retrieve_polarity_score(Italian_train)


# In[88]:


get_top_words(Italian_Cuisine_polarity_score, 'positive',30)


# In[89]:


get_top_words(Italian_Cuisine_polarity_score, 'negative',30)


# In[90]:


Italian_top_positive_words = ['delicious','fresh','friendly','recommend','reasonable','authentic',
                             'attentive','fun','refreshing','classic']
Italian_top_negative_words = ['cold','hard','wrong','bland','expensive','slow','greasy','fried','frozen','dirty']
Italian_top_words = Italian_Cuisine_polarity_score.loc[Italian_top_positive_words+Italian_top_negative_words,'polarity']
plot_top_words(Italian_top_words,'Italian')


# ### Combine all top words to compare among different cuisine typies

# In[91]:


# In[45]:


all_Cuisine_category = {'cuisine':['Korean','Japanese','Chinese','Thai','Vietnamese','French','Italian']}
cuisine_positive_words = pd.DataFrame(all_Cuisine_category)
for i,word in enumerate(Korean_top_positive_words):
    cuisine_positive_words.loc[0,i] = word



for i,word in enumerate(Korean_top_positive_words):
    cuisine_positive_words.iloc[0,i] = word
for i,word in enumerate(Japanese_top_positive_words):
    cuisine_positive_words.iloc[1,i] = word
for i,word in enumerate(Chinese_top_positive_words):
    cuisine_positive_words.iloc[2,i] = word
for i,word in enumerate(Thai_top_positive_words):
    cuisine_positive_words.iloc[3,i] = word
for i,word in enumerate(Viet_top_positive_words):
    cuisine_positive_words.iloc[4,i] = word
for i,word in enumerate(French_top_positive_words):
    cuisine_positive_words.iloc[5,i] = word
for i,word in enumerate(Italian_top_positive_words):
    cuisine_positive_words.iloc[6,i] = word

cuisine_positive_words.drop(9,axis=1,inplace=True)
cuisine_positive_words.columns=['0','1','2','3','4','5','6','7','8','9']
cuisine_positive_words['cuisine']=['Korean','Japanese','Chinese','Thai','Vietnamese','French','Italian']
cuisine_positive_words.set_index('cuisine', inplace=True)



all_Cuisine_category = {'cuisine':['Korean','Japanese','Chinese','Thai','Vietnamese','French','Italian']}
cuisine_negative_words = pd.DataFrame(all_Cuisine_category)
for i,word in enumerate(Korean_top_negative_words):
    cuisine_negative_words.loc[0,i] = word



for i,word in enumerate(Korean_top_negative_words):
    cuisine_negative_words.iloc[0,i] = word
for i,word in enumerate(Japanese_top_negative_words):
    cuisine_negative_words.iloc[1,i] = word
for i,word in enumerate(Chinese_top_negative_words):
    cuisine_negative_words.iloc[2,i] = word
for i,word in enumerate(Thai_top_negative_words):
    cuisine_negative_words.iloc[3,i] = word
for i,word in enumerate(Viet_top_negative_words):
    cuisine_negative_words.iloc[4,i] = word
for i,word in enumerate(French_top_negative_words):
    cuisine_negative_words.iloc[5,i] = word
for i,word in enumerate(Italian_top_negative_words):
    cuisine_negative_words.iloc[6,i] = word

cuisine_negative_words.drop(9,axis=1,inplace=True)
cuisine_negative_words.columns=['0','1','2','3','4','5','6','7','8','9']
cuisine_negative_words['cuisine']=['Korean','Japanese','Chinese','Thai','Vietnamese','French','Italian']
cuisine_negative_words.set_index('cuisine', inplace=True)


cuisine_positive_words

cuisine_negative_words



# In[46]:


cuisine_positive_words


# In[ ]:




