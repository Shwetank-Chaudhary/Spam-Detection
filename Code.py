import nltk
pip install nltk.download("stopwords")
import pandas as pd
import nltk.corpus.stopwords
from sklearn.naive_bayes import MultinomailNB()
import string
from sklearn.feature_extraction import TfidfTransformer,CounteVectorizer

spamfile=pd.reead_csv("https://raw.githubusercontent.com/w1449550206/Spam-classification/master/SMSSpamCollection.txt",sep='/t',names=['response','message'])
spamfile.head()
