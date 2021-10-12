#!/usr/bin/env python
# coding: utf-8
    
# ## Sentiment Analysis on Amazon Alexa
    
# ## Importing all Required Libraries
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins,fig_to_html
import seaborn as sns
import re #regular expression
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
     
def main():       
    # # Reading data set 
    data = pd.read_csv("amazon_alexa.csv")
    data.rename(columns={'verified_reviews': 'text'},inplace=True)
    data['length'] = data['text'].apply(len)
    data1 = data.groupby('length').describe().sample(10)
    
    # ## Data Visualisation    
    # color = plt.cm.copper(np.linspace(0, 1, 15))
    # fig1 = data['variation'].value_counts().plot.bar(color = color, figsize = (10, 6))
    # plt.title('Distribution of Variations in Alexa', fontsize = 15)
    # plt.xlabel('variations')
    # plt.ylabel('count')
    # plt.savefig("static/variationplot.png")
    # plt.show()
            
    # ## Plot the Rating Column
    # rating_dist_plot = sns.distplot(data['rating'])
    # rating_dist_plot.set_title("Distribution plot of Ratings")
    
    # rating_count_plot=sns.countplot(x='rating', data=data)
    # rating_count_plot.set_title("Countplot of Ratings")
    # plt.savefig("static/ratingcountplot.png")
    
    #Distribution of Ratings for Alexa
    ratings = data["rating"].value_counts()
    numbers = ratings.index
    quantity = ratings.values
    
    custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
    plt.figure(figsize=(7, 7))
    plt.pie(quantity, labels=numbers, autopct='%1.0f%%', colors=custom_colors)
    central_circle = plt.Circle((0, 0), 0.5, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(central_circle)
    plt.rc('font', size=12)
    plt.title("Amazon Alexa Reviews", fontsize=20)
    plt.savefig("static/ratingpieplot.png")
    plt.show()
    
    
    Totalrating=data["rating"].count()
    Totalrating
    onestar=data.loc[data['rating'] == 1].count()
    onestar[0]
    twostar=data.loc[data['rating'] == 2].count()
    twostar[0]
    threestar=data.loc[data['rating'] == 3].count()
    threestar[0]
    fourstar=data.loc[data['rating'] == 4].count()
    fourstar[0]
    fivestar=data.loc[data['rating'] == 5].count()
    fivestar[0]
    rtotal=data['rating'].loc[data['rating']].count()
    
    d=dict();
    d[' 1 : Count for 1 * Rating ']=onestar[0]
    d[' 2 : Count for 2 * Rating ']=twostar[0]
    d[' 3 : Count for 3 * Rating ']=threestar[0]
    d[' 4 : Count for 4 * Rating ']=fourstar[0]
    d[' 5 : Count for 5 * Rating ']=fivestar[0]
    d[' 6 : Count of Total Rating']=rtotal
    
    return(d)

def review(data):
    # # # EDA for categories 0 and 1
    # negativefeedback=data.loc[data['feedback'] == 0].count()
    # negativefeedback
    # postivefeedback=data.loc[data['feedback'] == 1].count()
    # postivefeedback
    # feedback_plot=sns.countplot(x='feedback', data=data)
    # feedback_plot.set_title("Countplot for Feedbacks")
    # plt.savefig("static/feedbackcountplot.png")
    # # # Data Cleaning
    
    def clean_text(text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub("[0-9" "]+"," ",text)
        text = re.sub('[‘’“”…]', '', text)
        return text
    
    clean = lambda x: clean_text(x)
    data['text'] = data.text.apply(clean)
    
    #Word frequency
    freq = pd.Series(' '.join(data['text']).split()).value_counts()[:20] # for top 20    
    
    #removing stopwords
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    new_data=data['text']
    
    #word frequency after removal of stopwords
    freq_Sw = pd.Series(' '.join(data['text']).split()).value_counts()[:20] # for top 20
    
    # count vectoriser tells the frequency of a word.    
    vectorizer = CountVectorizer(min_df = 1, max_df = 0.9)
    X = vectorizer.fit_transform(data["text"])
    word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
    word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])

    #TFIDF - Term frequency inverse Document Frequency
    vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True) #keep top 1000 words
    doc_vec = vectorizer.fit_transform(data["text"])
    names_features = vectorizer.get_feature_names()
    dense = doc_vec.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns = names_features)   
    
    # #Bi-gram
    # def get_top_n2_words(corpus, n=None):
    #     vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
    #             max_features=2000).fit(corpus)
    #     bag_of_words = vec1.transform(corpus)
    #     sum_words = bag_of_words.sum(axis=0) 
    #     words_freq = [(word, sum_words[0, idx]) for word, idx in     
    #                   vec1.vocabulary_.items()]
    #     words_freq =sorted(words_freq, key = lambda x: x[1], 
    #                 reverse=True)
    #     return words_freq[:n]
    # top2_words = get_top_n2_words(data["text"], n=200) #top 200
    # top2_df = pd.DataFrame(top2_words)
    # top2_df.columns=["Bi-gram", "Freq"]
   
    # #Bi-gram plot    
    # top20_bigram = top2_df.iloc[0:20,:]
    # fig = plt.figure(figsize = (10, 5))
    # plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
    # plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])
    
    # #Tri-gram
    # def get_top_n3_words(corpus, n=None):
    #     vec1 = CountVectorizer(ngram_range=(3,3), 
    #            max_features=2000).fit(corpus)
    #     bag_of_words = vec1.transform(corpus)
    #     sum_words = bag_of_words.sum(axis=0) 
    #     words_freq = [(word, sum_words[0, idx]) for word, idx in     
    #                   vec1.vocabulary_.items()]
    #     words_freq =sorted(words_freq, key = lambda x: x[1], 
    #                 reverse=True)
    #     return words_freq[:n]
    
    # top3_words = get_top_n3_words(data["text"], n=200)
    # top3_df = pd.DataFrame(top3_words)
    # top3_df.columns=["Tri-gram", "Freq"]
    
    # #Tri-gram plot
    # top20_trigram = top3_df.iloc[0:20,:]
    # fig = plt.figure(figsize = (10, 5))
    # plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
    # plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])
     
    # Most frequently occuring words - Top 20    
    cv = CountVectorizer(stop_words = 'english')
    words = cv.fit_transform(data.text)
    sum_words = words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    
    # plt.style.use('fivethirtyeight')
    # color = plt.cm.viridis(np.linspace(0, 1, 20))
    # frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
    # plt.title("Most Frequently Occurring Words - Top 20")
    # frequently_occu_words_plot = plt.show()
    
    # # WordCloud
    string_Total = " ".join(new_data)
   # Define a function to plot word cloud
    def plot_cloud(wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");
    stopwords = STOPWORDS
    stopwords.add('will')
    stopwords.add('im')
    stopwords.add('one') # beacuse everyone using this in context of the item ie this one or buy one etc
    wordcloud = WordCloud(width = 2000, height = 1000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(string_Total)
    # Plot
    plot_cloud(wordcloud)
    wordcloud.to_file("static/wordcloud.png")
    
    data_new = data.drop(["variation", "date","rating"], axis = 1)
    vectorizer = CountVectorizer(stop_words='english',max_features=1000, binary=True)
    all_features = vectorizer.fit_transform(data_new.text)
    # all_features.shape
    # vectorizer.vocabulary_
    
    review_train,review_test = train_test_split(data_new,test_size=0.3)
    X_train=review_train.text
    X_test =review_test.text
    y_train=review_train.feedback
    y_test=review_test.feedback
    X_train_vect = vectorizer.fit_transform(X_train)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    
    # ## Train dataset balancing
    
    sm = SMOTE()
    X_train_res, y_train_res = sm.fit_resample(X_train_vect, y_train)
    unique, counts = np.unique(y_train_res, return_counts=True)
    print(list(zip(unique, counts)))
    # y_train_res
    # X_train_res
    
    # ## Model Building
    
    # ## NAIVE BAYES CLASSIFIER
    
    # Create the classifier and fit it to our training data
    nb = MultinomialNB()
    model_NB = nb.fit(X_train_res, y_train_res)
    #nb.score(X_train_res, y_train_res)
    train_pred_NB = model_NB.predict(X_train_res)
    
    dNB_train=dict();            
    dNB_train['Result for training data using Naive Bayes Classifier : ']=' '
    dNB_train['Accuracy  : ']=accuracy_score(y_train_res, train_pred_NB) * 100
    dNB_train['Recall  : ']=recall_score(y_train_res, train_pred_NB) * 100
    dNB_train['Precision  : ']=precision_score(y_train_res, train_pred_NB) * 100
    dNB_train['F1-Score : ']=f1_score(y_train_res, train_pred_NB) * 100
    print("NB_train",dNB_train)

    # ## RANDOM FOREST CLASSIFIER
    
    # # Create the classifier and fit it to our training data
    # model_RF = RandomForestClassifier(random_state=7, n_estimators=100)
    # model_RF.fit(X_train_res, y_train_res)
    # # Predict classes given the validation features
    # train_pred_RF = model_RF.predict(X_train_res)
    
    # dRF_train=dict();            
    # dRF_train['Result for training data using Random Forest Classifier : '] =' '
    # dRF_train['Accuracy : ']=accuracy_score(y_train_res, train_pred_RF) * 100
    # dRF_train['Recall  : ']=recall_score(y_train_res, train_pred_RF) * 100
    # dRF_train['Precision  : ']=precision_score(y_train_res, train_pred_RF) * 100
    # dRF_train['F1-Score  : ']=f1_score(y_train_res, train_pred_RF) * 100
    # print("RF_train",dRF_train)
    
    
    # # ## LOGISTIC REGRESSION CLASSIFIER
    
    # # Create the classifier and fit it to our training data
    # LR = LogisticRegression()
    # model_LR = LR.fit(X_train_res, y_train_res)
    # # Predict classes given the validation features
    # train_pred_LR = model_LR.predict(X_train_res)
    
    # dLR_train=dict();            
    # dLR_train['Result for training data using Logistic Regression Classifier : '] =' '
    # dLR_train['Accuracy : ']=accuracy_score(y_train_res, train_pred_LR) * 100
    # dLR_train['Recall  : ']=recall_score(y_train_res, train_pred_LR) * 100
    # dLR_train['Precision  : ']=precision_score(y_train_res, train_pred_LR) * 100
    # dLR_train['F1-Score  : ']=f1_score(y_train_res, train_pred_LR) * 100
    # print("LR_train",dLR_train)
    
    # # Create table for all three classifiers result    
    # data_train = {'Naive Bayes': [accuracy_score(y_train_res, train_pred_NB) * 100, recall_score(y_train_res, train_pred_NB) * 100, 
    #                         precision_score(y_train_res, train_pred_NB)*100, f1_score(y_train_res, train_pred_NB) * 100], 
    #         'Logistic Regression': [accuracy_score(y_train_res, train_pred_LR) * 100, recall_score(y_train_res, train_pred_LR) * 100,
    #                                 precision_score(y_train_res, train_pred_LR) * 100, f1_score(y_train_res, train_pred_LR) * 100], 
    #         'Random Forest' :[accuracy_score(y_train_res, train_pred_RF) * 100,recall_score(y_train_res, train_pred_RF) * 100,
    #                          precision_score(y_train_res, train_pred_RF) * 100,f1_score(y_train_res, train_pred_RF) * 100]}
    
    # train_result_df = pd.DataFrame(data_train,index = ['Accuracy','Recall Score','Precision Score','F1 Score Score'])
    # print("Result for Train data :",train_result_df)
     
    #RESULTS FOR TESTING DATA
    X_test_vect = vectorizer.transform(X_test)
    y_pred = nb.predict(X_test_vect)
    
    dNB_test=dict();            
    dNB_test['Result for testing data using Naive Bayes Classifier : '] =' '
    dNB_test['Accuracy  : ']=accuracy_score(y_test, y_pred) * 100
    dNB_test['Recall  : ']=recall_score(y_test, y_pred) * 100
    dNB_test['Precision  : ']=precision_score(y_test, y_pred) * 100
    dNB_test['F1-Score : ']=f1_score(y_test, y_pred) * 100
    print("NB_test",dNB_test)
       
    # # ## Testing for test dataset using Logistic Regression    
    # X_test_vect = vectorizer.transform(X_test)
    # test_pred_LR = model_LR.predict(X_test_vect)    
    # #RESULTS FOR TESTING DATA
    # dLR_test=dict();            
    # dLR_test['Result for testing data using Logistic Regression Classifier : '] =' '
    # dLR_test['Accuracy  : ']=accuracy_score(y_test, test_pred_LR) * 100
    # dLR_test['Recall  : ']=recall_score(y_test, test_pred_LR) * 100
    # dLR_test['Precision  : ']=precision_score(y_test, test_pred_LR) * 100
    # dLR_test['F1-Score : ']=f1_score(y_test, test_pred_LR) * 100
    # print("LR_test",dLR_test)
    
    
    # # ## Testing for test dataset using Random Forest    
    # X_test_vect = vectorizer.transform(X_test)
    # test_pred_RF = model_RF.predict(X_test_vect)  
    # #RESULTS FOR TESTING DATA
    # dRF_test=dict();            
    # dRF_test['Result for testing data using Random Forest Classifier : '] =' '
    # dRF_test['Accuracy  : ']=accuracy_score(y_test, test_pred_RF) * 100
    # dRF_test['Recall  : ']=recall_score(y_test, test_pred_RF) * 100
    # dRF_test['Precision  : ']=precision_score(y_test, test_pred_RF) * 100
    # dRF_test['F1-Score : ']=f1_score(y_test, test_pred_RF) * 100
    # print("RF_test",dRF_test)
    
    
    # # ## Camparison of Result for Test Data   
    # data_test = {'Naive Bayes': [accuracy_score(y_test, y_pred) * 100, recall_score(y_test, y_pred) * 100, 
    #                         precision_score(y_test, y_pred)*100, f1_score(y_test, y_pred) * 100], 
    #         'Logistic Regression': [accuracy_score(y_test, test_pred_LR) * 100, recall_score(y_test, test_pred_LR) * 100,
    #                                 precision_score(y_test, test_pred_LR) * 100, f1_score(y_test, test_pred_LR) * 100], 
    #         'Random Forest' :[accuracy_score(y_test, test_pred_RF) * 100,recall_score(y_test, test_pred_RF) * 100,
    #                          precision_score(y_test, test_pred_RF) * 100,f1_score(y_test, test_pred_RF) * 100]}
    
    # test_result_df = pd.DataFrame(data_test,index = ['Accuracy','Recall Score','Precision Score','F1 Score Score'])
    # print("Result for Test data : ",test_result_df)
    
    
    # ## Testing for overall dataset using Naive Bayes Classifier    
    X_overall_vect = vectorizer.transform(data.text)
    y_pred_overall = nb.predict(X_overall_vect)        
    dNB_overall=dict();            
    dNB_overall['Result for overall data using Naive Bayes Classifier : '] =' '
    dNB_overall['Accuracy  : ']=accuracy_score(data.feedback, y_pred_overall) * 100
    dNB_overall['Recall  : ']=recall_score(data.feedback, y_pred_overall) * 100
    dNB_overall['Precision  : ']=precision_score(data.feedback, y_pred_overall) * 100
    dNB_overall['F1-Score : ']=f1_score(data.feedback, y_pred_overall) * 100
    print("NB_overall",dNB_overall)
    
    # # we have to check this for custom values 
    
    a=["bad and worst device","alexa is good and best","bad device","loved it!","it was terrible!"]
    tdm = vectorizer.transform(a)
    pred_F = nb.predict(tdm)    
    print(pred_F)
    
    
    return()