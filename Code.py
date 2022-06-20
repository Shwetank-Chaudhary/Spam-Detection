import nltk
pip install nltk.download("stopwords")
import pandas as pd
import nltk.corpus.stopwords
from sklearn.naive_bayes import MultinomailNB()
import string
from sklearn.feature_extraction import TfidfTransformer,CounteVectorizer

spamfile=pd.reead_csv("https://raw.githubusercontent.com/w1449550206/Spam-classification/master/SMSSpamCollection.txt",sep='/t',names=['response','message'])
spamfile.head()

def message_text_process(m):
  message=[c for c in m if c not in string.punctutation]
  message=''.join(message)
  return [c for c in message.split() if c.lower() not in stopwords.words('english')]

words=CountVectorizer(analyzer=message_tet_process).fit(spamfile.message)
message=words.transform(spamfile['message'])

tfidf_transformer=TfidfTransformer().fit(message)
message_tfidf=tfidf_transformer.transform(message)
print(message_tfidf.shape)

spam_detect_model=MultinomialNB().fit(message,spamfie['response'])

//Checking the working of spam detector

m1=spamfile.message[4]
m1_vectorised=words.transform(m1)
m1_tfidf=tfidf_transformer.transform(m1_vectorised)

print("Value from the model: ",spam_detect_model.predict(m1_tfidf))
print("Actual Value: ",spamfile.response[4])
