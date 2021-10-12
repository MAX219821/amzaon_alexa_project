#!/usr/bin/env python
# coding: utf-8

# # Reading data set 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re #regular expression
import string
import pickle

def rating():
    data = pd.read_csv("amazon_alexa.csv")
    
    
    data.rename(columns={'verified_reviews': 'text'},inplace=True)
    

    
    data.isnull().sum()
    
   
    Totalrating=data["rating"].count()
     
    
    onestar=data.loc[data['rating'] == 1].count()
    
    
    twostar=data.loc[data['rating'] == 2].count()
    
      
    
    threestar=data.loc[data['rating'] == 3].count()
    
    
    
    fourstar=data.loc[data['rating'] == 4].count()
    
    
    fivestar=data.loc[data['rating'] == 5].count()
    
    
    d=dict();
    d[' 1 : Count of 1 star rating']=onestar[0]
    d[' 2 : Count of 2 star rating'] = twostar[0]
    d[' 3 : Count of 3 star rating']=threestar[0]
    d[' 4 : Count of 4 star rating']=fourstar[0]
    d[' 5 : Count of 5 star rating']=fivestar[0]
    d[' 6 : Count of Total rating']=Totalrating

    return(d)

def clean_text(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("[0-9" "]+"," ",text)
    text = re.sub('[‘’“”…]', '', text)
    return text


def reviewfunc(inputvalue):

    data = pd.read_csv("amazon_alexa.csv")
                
    data.rename(columns={'verified_reviews': 'text'},inplace=True)
       
    clean = lambda x: clean_text(x)
     
    data['text'] = data.text.apply(clean)
    data.text
    

    freq = pd.Series(' '.join(data['text']).split()).value_counts()[:20] # for top 20
    freq
    
    
    #removing stopwords
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    new_data=data['text']
    

    #word frequency after removal of stopwords
    freq_Sw = pd.Series(' '.join(data['text']).split()).value_counts()[:20] # for top 20
    freq_Sw
    
    
    string_Total = " ".join(new_data)
    

    import matplotlib.pyplot as plt
    # get_ipython().run_line_magic('matplotlib', 'inline')
    from wordcloud import WordCloud, STOPWORDS
    # Define a function to plot word cloud
    def plot_cloud(wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");
    
    
    # In[20]:
    
    
    stopwords = STOPWORDS
    stopwords.add('will')
    stopwords.add('im')
    stopwords.add('one') # beacuse everyone using this in context of the item ie this one or buy one etc
    wordcloud = WordCloud(width = 2000, height = 1000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(string_Total)
    # Plot
    plot_cloud(wordcloud)
    
    
    
    
    data.drop(["variation", "date","rating"], axis = 1, inplace = True)
    
    
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words='english',max_features=3000, binary=True)
    all_features = vectorizer.fit_transform(data.text)
    all_features.shape
    
    
    # In[25]:
    
    
    vectorizer.vocabulary_
    
    
    # In[26]:
    
    
    from sklearn.model_selection import train_test_split
    review_train,review_test = train_test_split(data,test_size=0.3)
    X_train=review_train.text
    X_test =review_test.text
    y_train=review_train.feedback
    y_test=review_test.feedback
    X_train_vect = vectorizer.fit_transform(X_train)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    
    # ## Train dataset balancing
    
    # In[27]:
    
    
    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    X_train_res, y_train_res = sm.fit_resample(X_train_vect, y_train)
    unique, counts = np.unique(y_train_res, return_counts=True)
    print(list(zip(unique, counts)))
    y_train_res
    X_train_res
    
    
    # ## Model Building
    
    # In[28]:
    
    
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train_res, y_train_res)
    nb.score(X_train_res, y_train_res)
    train_pred = nb.predict(X_train_res)
    train_pred
    
    
    # In[29]:
    
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    print("Accuracy: {:.2f}%".format(accuracy_score(y_train_res, train_pred) * 100))


    # In[30]:
    
    
    X_test_vect = vectorizer.transform(X_test)
    y_pred = nb.predict(X_test_vect)
    y_pred
    
    
    # In[31]:
    
    
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    
    
    # In[32]:
    
    
    ## Testing for overall dataset
    
    
    # In[33]:
    
    
    X_overall_vect = vectorizer.transform(data.text)
    y_pred_overall = nb.predict(X_overall_vect)
    y_pred_overall
    
    
    # In[34]:
    
    
    print("Accuracy: {:.2f}%".format(accuracy_score(data.feedback, y_pred_overall) * 100))
    
    a = sum(y_pred_overall==0)
    b = sum(y_pred_overall==1)
    c = len(y_pred_overall)
    
    d1=dict();
    d1[' 1 : Positive feedback ']=b
    d1[' 2 : Negative Feedback ']=a
    d1[' 3 : Total feddback ']=c
    
    
    userinput = [inputvalue]
    if inputvalue is not None:
        tdm = vectorizer.transform(userinput)
        pred_F = nb.predict(tdm)
        print(pred_F)
        predicted_data = int(pred_F[0])
        if predicted_data ==0:
            message="Negative Sentiment"
        elif predicted_data ==1:
            message="Positive Sentiment"
            
        
    else:
        message="No value"
        
    d2=dict();
    d2[' 1 :User input ']=userinput
    d2[' 2 : Sentiment Value ']=message         
    
            
    return(d1,d2)


    

