# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:24:46 2016

@author: Eng.Alla Khaled
"""
"""to get the 20 words and their frequency percentage 
with highest frequency in an English Wikipedia article. 
applications are recommender systems, chatbots and NLP, sentiment analysis,
data visualization,
market research"""
# import dependencies
from bs4 import BeautifulSoup
import requests # for pulling pushing and authenticating from the web
import re  # regular expression, make a special text string for describing a search pattern
import operator # export a set of effecient functions correspomding to the intrinsic operators comparsion, greater than, less than,..
import json # data format
from tabulate import tabulate # take a list of lists and transform it into table
import sys # system calls, deal with user arguments
from stop_words import get_stop_words # stop words like the, at, to,...,,,you definetly want to eleiminate it

#get the words
def getWorldList(url):
    word_list =[]
    source_code = requests.get(url) #raw data
    plain_text = source_code.text #convert to text
    soup = BeautifulSoup(plain_text, 'lxml') #lxml format
    
    #find the words in paragraph tag
    for text in soup.findAll('p'):
        if text.text is None:
            continue
        content = text.text # content
        words = content.lower().split() #lowercase and split into an array
        
        for word in words:
            cleaned_word = clean_word(word)
            #if there is still something there
            if len(cleaned_word) > 0:
                word_list.append(cleaned_word) #add it to our word list
    return word_list

#clean word with regex
def clean_word(word):
    cleaned_word = re.sub('[^A-Za-z]+', '', word)
    return cleaned_word

def createFrequencyTable(word_list):
    word_count = {} # word count
    for word in word_list:
        if word in word_count: # index is the word
            word_count[word] += 1
        else:
           word_count[word] = 1
    return word_count
       
def remove_stop_words(frequency_list):
    stop_words = get_stop_words('en') # get stop words in the English language
    
    temp_list = []
    for key, value in frequency_list:
        if key not in stop_words:
            temp_list.append([key, value])
    return temp_list
    

"""
#get data from wikipedia api
# Use any programming language to make an HTTP GET request for that URL 
# (or just visit that link in your browser), and you'll get a JSON document
# which includes the current wiki markup for the page titled "Main Page".
# Changing the format to jsonfm will return a "pretty-printed" HTML result 
# good for debugging."""

# access wiki API. json format. query it for data. search type. shows list of possibilities, srsearch will take the search word
wikipedia_api_link = "https://en.wikipedia.org/w/api.php?format=json&action=query&list=search&srsearch="
wikipedia_link = "https://en.wikipedia.org/wiki/"

# a user query will look like this:
# python main.py batman yes /*which means to query the 2o most frequent words in batman articles, and yes means remove stop words*\
# batman -> is the search query in the previous example

# if the search word is too small throw an error
if (len(sys.argv) < 2):
    print("Enter valid string")
    exit()
    
#get the search word
string_query = sys.argv[1]

#to remove stop words or not
if(len(sys.argv) > 2):
    search_mode = True
else:
    search_mode = False
    
# create our url
url = wikipedia_api_link + string_query

try:
    response = requests.get(url) # get the wiki article in resonse variable
    data = json.loads(response.content.decode("utf-8")) # load data from the article ansd decode it in the right format 'utf-8'
    
    # format this data
    wikipedia_page_tag = data['query']['search'][0]['title'] # storre the first link
    
    #create our new url
    url = wikipedia_link + wikipedia_page_tag
    page_world_list = getWorldList(url) # get list of words from the list
    # create a table of word counts
    page_word_count = createFrequencyTable(page_world_list) # create table of word counts
    sorted_word_frequency_list = sorted(page_word_count.items(), key=operator.itemgetter(1), reverse=True)     #sort the table by the frequency count
    #remove the stop words
    if(search_mode):
        sorted_word_frequency_list = remove_stop_words(sorted_word_frequency_list)
        
    # sum the total words to calkculate the frequency
    total_words_sum = 0
    for key, value in sorted_word_frequency_list:
        total_words_sum = total_words_sum + value
        
    # just get thw top 20 words
    if len(sorted_word_frequency_list) > 20:
        sorted_word_frequency_list = sorted_word_frequency_list[:20]
        
    # crete oour final list, words / frequency / percentage
    final_list = []
    for key, value in sorted_word_frequency_list:
        percentage_value = float(value) / total_words_sum
        final_list.append([key, value, round(percentage_value, 4)])
    
    #headers before the table    
    print_headers = ['Word','Frequency','Frequency Percentage']
    #print the table with tabulate
    print(tabulate(final_list, headers=print_headers, tablefmt='orgtbl'))
    
#throw an exception in case the connection breaks
except requests.exceptions.Timeout:
    print("The server didn't respond. please, try again later.")

    
    
  