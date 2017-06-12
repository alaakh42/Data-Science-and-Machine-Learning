# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 11:31:43 2016

@author: Eng.Alaa Khaled
"""

"""Urllib2: It is a Python module which can be used for fetching URLs. 
It defines functions and classes to help with URL actions (basic and digest
 authentication, redirections, cookies, etc).
"""

"""BeautifulSoup: It is an incredible tool for pulling out
 information from a webpage. You can use it to extract tables,
 lists, paragraph and you can also put filters to extract
 information from web pages. 
"""

"""BeautifulSoup does not fetch the web page for us.
 That’s why, we use urllib2 in combination with the BeautifulSoup library."""
 
# Python has several other options for HTML scraping in addition to BeatifulSoup. Here are some others:
#mechanize
#scrapemark
#scrapy
 
import urllib2 as url

#specify the url
wiki = "https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India"

#query the website and return the html to the variable 'page'
page = url.urlopen(wiki)

#import the Beautifuk soup functions to parse the data returned from the website
from bs4 import BeautifulSoup

#parse the html in 'page' varaible, and store it in  Beautiful Soup format
soup = BeautifulSoup(page)

print soup.prettify() #take a look of the nested structure of the HTML page

#pay around and discover the different tages using 
#soup.<tag_name> e.g. soup.title.string [to return data within the tag as a string]

# to extract all links in the HTML page <a>
all_links = soup.find_all("a")
for link in all_links:
    print link.get("href")

# we are seeking a table to extract information about state 
#capitals, we should identify the right table first
all_tables = soup.find_all('table')
# filter the tables 
right_table = soup.find('table', class_='wikitable sortable plainrowheaders') #to figure out the class name, use 'Inspect element'

#Extract the information to DataFrame: Here, we need to 
#iterate through each row (tr) and then assign each element of
#tr (td) to a variable and append it to a list.
# Ignore <th> table heading

#Generate lists
A=[]
B=[]
C=[]
D=[]
E=[]
F=[]
G=[]

for row in right_table.findAll("tr"):
    cells = row.findAll('td')
    states = row.findAll('th') #to store the second column data
    if len(cells) == 6: #only extract table body no heading
        A.append(cells[0].find(text=True)) # to access value of each element, we will use “find(text=True)” option with each element
        B.append(states[0].find(text=True))
        C.append(cells[1].find(text=True))
        D.append(cells[2].find(text=True))
        E.append(cells[3].find(text=True))
        F.append(cells[4].find(text=True))
        G.append(cells[5].find(text=True))
        
#to convert lists into data frames
import pandas as pd
df = pd.DataFrame(A, columns = ['Number'])
df['State/UT'] = B
df['Admin_Capital'] = C
df['Legislative_Capital'] = D
df['Judiciary_Capital']= E
df['Year_Capital']= F
df['Former_Capital']= G

"""Similarly, you can perform various other types of web scraping 
   using “BeautifulSoup“. This will reduce your manual efforts to 
   collect data from web pages. You can also look at the other 
   attributes like .parent, .contents, .descendants and .next_sibling, 
   .prev_sibling and various attributes to navigate using tag name.
   These will help you to scrap the web pages effectively.-"""


