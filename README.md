# ComprehensiveStudy_UserSentiment

Steps to run the project:

Install Jupyter notebook using the command "pip install jupyterlab" 

Download "yelp_academic_dataset_business.json" and "yelp_academic_dataset_review.json" dataset from the link https://www.kaggle.com/yelp-dataset/yelp-dataset

Download "Yelp_PredictSentiment.py" and "Yelp_SentimentPolarity.py " from the git repo https://github.com/akshay-26/ComprehensiveStudy_UserSentiment

(Sample files of both the dataset available here: https://github.com/akshay-26/ComprehensiveStudy_UserSentiment/blob/main/yelp_business.json,
 https://github.com/akshay-26/ComprehensiveStudy_UserSentiment/blob/main/yelp_review.json)

Import both the python files to jupyter notebook

Install the following libraries:

Library | Command
--- | ---
sklearn    | pip install -U scikit-learn
imblearn   | pip install imblearn
pandas     | pip install pandas
tqdm       | pip install tqdm
nltk       | pip install --user -U nltk
seaborn    | pip install seaborn
textblob   | pip install -U textblob
matplotlib | pip install matplotlib
spacy      | pip install -U pip setuptools wheel , pip install -U spacy
keras      | pip install tensorflow
scipy      | pip install --user scipy
gensim     | pip install gensim
worldcloud | pip install wordcloud


Import Yelp_PredictSentiment.py file in Jupyter Notebook and click on Cell -> Run All 

Import Yelp_SentimentPolarity.py file in Jupyter Notebook and click on Cell -> Run All 
