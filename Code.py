import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import string
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

spamfile=pd.read_csv("https://raw.githubusercontent.com/w1449550206/Spam-classification/master/SMSSpamCollection.txt",sep='\t',names=['response','message'])
spamfile.head()

def message_text_process(mess):
  message=[char for char in mess if char not in string.punctuation]
  message=''.join(message)
  return [c for c in message.split() if c.lower() not in stopwords.words('english')]

words=CountVectorizer(analyzer=message_text_process).fit(spamfile.message)
message=words.transform(spamfile['message'])

tfidf_transformer=TfidfTransformer().fit(message)
message_tfidf=tfidf_transformer.transform(message)

spam_detect_model=MultinomialNB().fit(message,spamfile['response'])

m1=spamfile['message'][4]
m1_vectorised=words.transform([m1])
m1_tfidf=tfidf_transformer.transform(m1_vectorised)

print("Value from the model: ",spam_detect_model.predict(m1_tfidf))
print("Actual Value: ",spamfile.response[4])
