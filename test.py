import pandas as pd
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

drug_name = []
condition = []
review = []
rating = []

dataset = pd.read_csv("Dataset/drugsComTrain_raw.tsv",sep="\t",nrows=5000)
print(np.unique(dataset['rating'],return_counts=True))
if os.path.exists('model/data.npy'):
    data = np.load("model/data.npy")
    drug_name = data[0]
    condition = data[1]
    review = data[2]
    rating = data[3]
else:
    for i in range(len(dataset)):
        dname = dataset.get_value(i,"drugName")
        cond = str(dataset.get_value(i,"condition"))
        reviewText = dataset.get_value(i,"review")
        ratings = dataset.get_value(i,"rating")
        reviewText = str(reviewText)
        reviewText = reviewText.strip().lower()
        reviewText = cleanPost(reviewText)
        drug_name.append(dname)
        condition.append(cond)
        review.append(cond.lower()+" "+reviewText)
        rating.append(ratings-1)
        print(i)
    data = [drug_name,condition,review,rating]
    data = np.asarray(data)
    drug_name = data[0]
    condition = data[1]
    review = data[2]
    rating = data[3]
    np.save("model/data",data)
if os.path.exists('model/tfidf.txt'):
    with open('model/tfidf.txt', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
    file.close()
    with open('model/X.txt', 'rb') as file:
        X = pickle.load(file)
    file.close()
else:
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=700)
    tfidf = tfidf_vectorizer.fit_transform(review).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    print(df)
    print(df.shape)
    df = df.values
    X = df[:, 0:df.shape[1]]
    with open('model/tfidf.txt', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    file.close()
    with open('model/X.txt', 'wb') as file:
        pickle.dump(X, file)
    file.close()

Y = rating
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
predict = lr.predict(X_test)
lr_precision = precision_score(y_test, predict,average='macro') * 100
lr_recall = recall_score(y_test, predict,average='macro') * 100
lr_fmeasure = f1_score(y_test, predict,average='macro') * 100
lr_acc = accuracy_score(y_test,predict)*100
print(str(lr_precision)+" "+str(lr_recall)+" "+str(lr_fmeasure)+" "+str(lr_acc))
    
