# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 20:41:20 2021

@author: jaces
"""
# Import libraries
import requests
import json
import numpy as np

from pymongo import MongoClient
from pprint import pprint


print('*********************************************************')
print('BEGIN')
print('*********************************************************')

print('*********************************************************')
print('** 01. Flexibly Structured Data')
print('*********************************************************')
print('** 01.01 Intro to MongoDB and the Nobel Prize dataset')
print('*********************************************************')
# Client connects to "localhost" by default
client = MongoClient()

# PREPARING THE DATABASE
# Use empty document {} as a filter
filter = {}
database_collections = {'nobel' : ['prizes', 'laureates'], 
                        'nobel2': ['nobelPrizes', 'laureates']}
for database in database_collections:
    db = client[database]
    for collection_name in database_collections[database]:
        db[collection_name].delete_many(filter)
print('Cleaning finished...')

# READING FROM API SERVICE
# Create local "nobel2" database on the fly
db = client["nobel2"]

years = np.arange(1901, 2020)
path_data = {"nobelPrizes": 'https://api.nobelprize.org/2.0/nobelPrizes', 
             "laureates": 'https://api.nobelprize.org/2.0/laureates'}
for nobelPrizeYear in years:
    for collection_name in path_data:
        # collect the data from the API
        response = requests.get(path_data[collection_name], params={'nobelPrizeYear': nobelPrizeYear})
        
        # convert the data to json
        documents = response.json()[collection_name]
    
        # Create collections on the fly
        if len(documents) > 0:
            db[collection_name].insert_many(documents) 
            print(f'Year {nobelPrizeYear} added...')

print('Loading finished...')


## Create local "nobel2" database on the fly
#db = client["nobel2"]
#
#path_data = {"nobelPrizes": 'https://api.nobelprize.org/2.0/nobelPrizes', 
#             "laureates": 'https://api.nobelprize.org/2.0/laureates'}
#for collection_name in path_data:
#    # collect the data from the API
#    response = requests.get(path_data[collection_name], params={'limit': 100})
#    
#    # convert the data to json
#    documents = response.json()[collection_name]
#    
#    # Create collections on the fly
#    db[collection_name].insert_many(documents)
#
#print('Loading finished...')

# READING FROM LOCAL FILE
# Create local "nobel" database on the fly
db = client["nobel"]

path_file = 'data/{}.json'

for collection_name in ['prizes', 'laureates']:
    documents= json.load(open(path_file.format(collection_name)))
    # Create collections on the fly
    db[collection_name].insert_many(documents)

print('Loading finished...')

# Count documents in a collection
# Use empty document {} as a filter
filter = {}

# Count documents in a collection
n_prizes = db.prizes.count_documents(filter)
n_laureates = db.laureates.count_documents(filter)

print('Documents in Prizes collections   :', n_prizes)
print('Documents in Laureates collections:', n_laureates)

print('*********************************************************')
print('** 01.02 Count documents in a collection')
print('*********************************************************')
# Create local "nobel" database on the fly
client = MongoClient()

print(client.nobel.prizes.count_documents({}))
print(client.nobel.laureates.count_documents({}))

print('*********************************************************')
print('** 01.03 Listing databases and collections')
print('*********************************************************')
# Save a list of names of the databases managed by client
db_names = client.list_database_names()
print(db_names)

# Save a list of names of the collections managed by the "nobel" database
nobel_coll_names = client.nobel.list_collection_names()
print(nobel_coll_names)

print('*********************************************************')
print('** 01.04 List fields of a document')
print('*********************************************************')
# Connect to the "nobel" database
db = client.nobel

# Retrieve sample prize and laureate documents
prize = db.prizes.find_one()
laureate = db.laureates.find_one()

# Print the sample prize and laureate documents
pprint(prize)
pprint(laureate)
print(type(laureate))

# Get the fields present in each type of document
prize_fields = list(prize.keys())
laureate_fields = list(laureate.keys())

print(prize_fields)
print(laureate_fields)

print('*********************************************************')
print('** 01.05 Finding documents')
print('*********************************************************')
# Filters as (sub)documents
filter_doc = {'born': '1845-03-27',
              'diedCountry': 'Germany',
              'gender': 'male',
              'surname': 'Röntgen'}

print('Filter:', filter_doc)
print(db.laureates.count_documents(filter_doc))

# Simple filters
for filter in [{'gender': 'female'}, {'diedCountry': 'France'}, {'bornCity': 'Warsaw'}]:
    print('Filter: {}\n{}'.format(filter, db.laureates.count_documents(filter)))
    
# Composing filters
filter_doc = {'gender': 'female',
              'diedCountry': 'France',
              'bornCity': 'Warsaw'}
print('Filter:', filter_doc)
print(db.laureates.count_documents(filter_doc))
pprint(db.laureates.find_one(filter_doc))

# Query operators
for filter in [{'diedCountry': {'$in': ['France', 'USA']}}, # Value in a range $in: <list>
               {'diedCountry': {'$ne': 'France'}}, # Not equal $ne : <value> 
               {'diedCountry': {'$gt': 'Belgium', '$lte': 'USA'}} #Comparison: > : $gt , ≥ : $gte; < : $lt , ≤ : $lte
              ]: 
    print('Filter: {}\n{}'.format(filter, db.laureates.count_documents(filter)))
    
print('*********************************************************')
print('** 01.06 "born" approximation')
print('*********************************************************')
print('Prior to 1800:', db.laureates.count_documents({'born': {'$lt':'1800'}}))
print('Prior to 1700:', db.laureates.count_documents({'born': {'$lt':'1700'}}))

print('*********************************************************')
print('** 01.07 Composing filters')
print('*********************************************************')
# Create a filter for laureates who died in the USA
criteria = {'diedCountry': 'USA'}
count = db.laureates.count_documents(criteria)
print(count)

# Create a filter for laureates who died in the USA but were born in Germany
criteria = {'diedCountry': 'USA', 
            'bornCountry': 'Germany'}
count = db.laureates.count_documents(criteria)
print(count)

# Create a filter for Germany-born laureates who died in the USA and with the first name "Albert"
criteria = {'diedCountry': 'USA', 
            'bornCountry': 'Germany',
            'firstname'  : 'Albert'}
count = db.laureates.count_documents(criteria)
print(count)

print('*********************************************************')
print("** 01.08 We've got options")
print('*********************************************************')
# Save a filter for laureates born in the USA, Canada, or Mexico
criteria = {'bornCountry': { "$in": ['USA', 'Canada', 'Mexico']}}
count = db.laureates.count_documents(criteria)
print(count)


# Save a filter for laureates who died in the USA and were not born there
criteria = {'bornCountry': { "$ne": 'USA'},
            'diedCountry': 'USA'}
count = db.laureates.count_documents(criteria)
print(count)

print('*********************************************************')
print('** 01.09 Dot notation: reach into substructure')
print('*********************************************************')
# A functional density
criteria = {"firstname": "Walter", "surname": "Kohn"}
pprint(db.laureates.find_one())

criteria = {"prizes.affiliations.name": ("University of California")}
print(criteria, db.laureates.count_documents(criteria))

criteria = {"prizes.affiliations.city": ("Berkeley, CA")}
print(criteria, db.laureates.count_documents(criteria))

criteria = {'surname': 'Naipaul'}
print(criteria, db.laureates.count_documents(criteria))

criteria = {"bornCountry": {"$exists": False}}
print(criteria, db.laureates.count_documents(criteria))

# Multiple prizes
criteria = {}
print('Total documents:', db.laureates.count_documents(criteria))

criteria = {"prizes": {"$exists": True}}
print('With prizes:', db.laureates.count_documents(criteria))

criteria = {"prizes.0": {"$exists": True}}
print('With at least one prize:', db.laureates.count_documents(criteria))

criteria = {"prizes.1": {"$exists": True}}
print('With more than one prize:', db.laureates.count_documents(criteria))

print('*********************************************************')
print('** 01.10 Choosing tools')
print('*********************************************************')
criteria = {'bornCountry': 'Austria', 
            'prizes.affiliations.country': {'$ne': 'Austria'}}
print(db.laureates.count_documents(criteria))

print('*********************************************************')
print('** 01.11 Starting our ascent')
print('*********************************************************')
# Filter for laureates born in Austria with non-Austria prize affiliation
criteria = {'bornCountry': 'Austria', 
            'prizes.affiliations.country': {"$ne": 'Austria'}}

# Count the number of such laureates
count = db.laureates.count_documents(criteria)
print(count)

print('*********************************************************')
print("** 01.12 Our 'born' approximation, and a special laureate")
print('*********************************************************')
criteria = {"born": "0000-00-00"}
print(criteria, db.laureates.count_documents(criteria))

# Filter for documents without a "born" field
criteria = {'born': {'$exists': False}}
count = db.laureates.count_documents(criteria)
print(criteria, count)

# Filter for laureates with at least three prizes
criteria = {"prizes.2": {'$exists': True}}
count = db.laureates.count_documents(criteria)
print(criteria, count)
doc = db.laureates.find_one(criteria)
pprint(doc)

print('*********************************************************')
print('END')
print('*********************************************************')