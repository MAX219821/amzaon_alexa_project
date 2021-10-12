# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 09:49:58 2021

@author: dell
"""
import os
from flask import Flask,render_template,request,session
from flask_session import Session
import secrets
import sqlite3

randomnumber=secrets.randbelow(10000000)
print(randomnumber)

app = Flask(__name__)
app.secret_key=str(randomnumber)
app.config['SESSION_TYPE'] ='filesystem'
Session(app)

@app.route('/')
def  Home():
    return render_template('Homepage.html')

@app.route('/Home')
def Mainpage():
    return render_template('mainpage.html')

@app.route('/', methods=["POST"])
def  signin():
    Name =request.form["Uname"]
    Passkey=request.form["email"]
    conn=sqlite3.connect("User.db")
    c=conn.cursor()        
    c.execute('SELECT COUNT(*) FROM  USER_DATA WHERE Email=?',(Passkey,))
    result=c.fetchall()
    conn.commit()
    conn.close()
    resultF=result[0][0]
    print('result',result)
    print('resultF',resultF)
    print(Passkey)
    if resultF==0:
        conn=sqlite3.connect("User.db")
        c=conn.cursor()
        c.execute('''INSERT INTO  USER_DATA(Name,Email) VALUES(?,?)''',(Name,Passkey))
        conn.commit()
        conn.close()
        Display="Welcome !!! you are new user added to our Data enthusiastic group "
    else:
        Display="Thank you for being our valuable Exsisting Supporter "
        
    
    return render_template('mainpage.html',Display=Display)


@app.route('/Rating')
def Rating():
    import Reviews_Classification_Naive_Bayes_Final
    Totalresults=Reviews_Classification_Naive_Bayes_Final.rating()
    rating=Totalresults 
    return render_template('ratingresult.html',rating=rating)


@app.route('/review')
def reviewtext():
    import Reviews_Classification_Naive_Bayes_Final
    inputvalue=None
    Totalresults=Reviews_Classification_Naive_Bayes_Final.reviewfunc(inputvalue)
    overallpred=Totalresults[0]
    # NBtrain=Totalresults[1]
    # NBtest=Totalresults[2]
    # NBoverall=Totalresults[3]
        
    return render_template('reviewresult.html',overallpred=overallpred)

@app.route('/tableau')
def Tableau():
    return render_template('Tableaudashboard.html')


@app.route('/customsentiment')
def  customsentiment():
    return render_template('customsentimentinput.html')


@app.route('/customsentiment', methods=["POST"])
def customsentimentF():
    inputvalue =request.form["sentence"]
    import Reviews_Classification_Naive_Bayes_Final
    customresultF=Reviews_Classification_Naive_Bayes_Final.reviewfunc(inputvalue)
    customresult=customresultF[1]
    return render_template('customsentimentresult.html',customresult=customresult)


    
   
if __name__ =="__main__":
    app.run()
