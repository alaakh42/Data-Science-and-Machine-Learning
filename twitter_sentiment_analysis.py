# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 20:53:23 2016

@author: Eng.Alaa Khaled
"""
"""
1- Tokenization: to take a tweet(s) and then divide it into words or small sentences called "tokens".
   those words represent the so called "bag of words Model"
2- Calculate the frequency of each word
3- Look at the sentiment value of each word from a sentiment lexicon that
   has all words sentiment values pre-recorded in values from -1:1
4- Then you classify the wordsof your tweet(s)


"""
import tweepy #library to access twitter api
from textblob import TextBlob # do the actual sentiment analysis
import csv

consumer_key = "aUqSzs0FSA69bd6L5YQTynWgn"
consumer_secret = "WMxn6abt133XYgTM1xbSC5oymkARo8CwrE5xIIwei3XiQkLI2M"

access_token = "2550873577-Jz3HUZESZD2J6QbNaSnIii7ZlPivNlSI1FUUB6r"
access_token_secret = "q79VwmUbhoQ1xRiMl0i58EyAtragmU2GDT1QN7YUIdghx"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
"""Now, the authentication to access the API is done"""

api = tweepy.API(auth) # using this api, we can do methods such as, Create Tweets, Delete Tweets, Find Twitter Users

public_tweets = api.search('Clinton') # collect tweets that contain 'trump'

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text) # calculate the polarity and subjectivity for each of those tweets
    print(analysis.sentiment)







#with open('sentiment.csv', 'w') as  f:
#    writer = csv.DictWriter(f, fieldnames=['Tweet', 'Sentiment'])
#    writer.writeheader()
#    for tweet in public_tweets:
#        text = tweet.text
#        cleanedtext = ' '.join([word for word in text.split(' ') if len(word) > 0 and word[0] != '@' 
#                      and word[0]!='#' and 'http' not in word and word != 'RT']) # clean tweets
#        cleanedtext = cleanedtext.encode('utf-8')
#        analysis = TextBlob(cleanedtext)
#        sentiment = analysis.sentiment.polarity
#        if sentiment >= 0:
#            polarity = 'Positive'
#        else:
#            polarity = 'Negative'
#        writer.writerow({'Tweet':text, 'Sentiment':polarity}) #print(cleanedtext, polarity)
