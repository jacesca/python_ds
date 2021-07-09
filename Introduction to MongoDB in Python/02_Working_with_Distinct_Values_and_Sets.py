# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:54:00 2021

@author: jaces
"""
# Import libraries
from pymongo import MongoClient
from pprint import pprint
from bson.regex import Regex

import re

# Client connects to "localhost" by default
client = MongoClient()

# Connect to "nobel" database on the fly
db = client["nobel"]

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 02 Working with Distinct Values and Sets')
print('*********************************************************')
print('** 02.01 Survey Distinct Values')
print('*********************************************************')
# An exceptional laureate
criteria = {"prizes.2": {"$exists": True}}
print(criteria, db.laureates.count_documents(criteria))
pprint(db.laureates.find_one(criteria))

# Using .distinct()
criteria = "gender"
print(f'\n{criteria}:', db.laureates.distinct(criteria))

# .distinct() with dot notation
criteria = "prizes.category"
print(f'\n{criteria}:', db.laureates.distinct(criteria))

print('*********************************************************')
print('** 02.02 Categorical data validation')
print('*********************************************************')
pprint(db.prizes.find_one())
pprint(db.laureates.find_one())

print(db.prizes.distinct("category"))
print(db.laureates.distinct("prizes.category"))

print('*********************************************************')
print('** 02.03 Never from there, but sometimes there at last')
print('*********************************************************')
# Countries recorded as countries of death but not as countries of birth
countries = set(db.laureates.distinct('diedCountry')) - set(db.laureates.distinct('bornCountry'))
print(countries)

print('*********************************************************')
print('** 02.04 Countries of affiliation')
print('*********************************************************')
# The number of distinct countries of laureate affiliation for prizes
bornCountry = len(db.laureates.distinct('bornCountry'))
diedCountry = len(db.laureates.distinct('diedCountry'))
affiliations = len(db.laureates.distinct('prizes.affiliations.country'))
print('bornCountry:', bornCountry)
print('diedCountry:', diedCountry)
print('prizes.affiliations.country:', affiliations)

print('*********************************************************')
print('** 02.05 Distinct Values Given Filters')
print('*********************************************************')
# Awards into prize shares
# Found a laureate document with a value of "4" for the "share" field in one of it's "prizes" subdocuments.
pprint(db.laureates.find_one({"prizes.share": "4"}))
pprint(db.prizes.find_one({"laureates.share": "4"}))

# High-share prize categories
print(db.laureates.distinct("prizes.category", {"prizes.share": '4'}))
print(db.prizes.distinct("category", {"laureates.share": "4"}),'\n')

# Prize categories with multi-winners
criteria = {"prizes.1": {"$exists": True}}
print(db.laureates.distinct("prizes.category", criteria))
for doc in db.laureates.find(criteria):
    for prize in doc['prizes']:
        print(prize['category'])
        
print('*********************************************************')
print('** 02.06 Born here, went there')
print('*********************************************************')
print(db.laureates.distinct('prizes.affiliations.country', {'bornCountry': 'USA'}))

print('*********************************************************')
print('** 02.07 Triple plays (mostly) all around')
print('*********************************************************')
# Save a filter for prize documents with three or more laureates
criteria = {"laureates.2": {'$exists': True}}

# Save the set of distinct prize categories in documents satisfying the criteria
triple_play_categories = set(db.prizes.distinct('category', criteria))
print(triple_play_categories)

all_categories = set(db.prizes.distinct('category'))
print(all_categories)
# Confirm literature as the only category not satisfying the criteria.
assert all_categories - triple_play_categories == {'literature'}

print('*********************************************************')
print('** 02.08 Filter Arrays using Distinct Values')
print('*********************************************************')
# Array fields and equality
criteria = {"prizes.category": "physics"}
print(criteria, db.laureates.count_documents(criteria))

# Array fields and equality, simplified
criteria = {'nicknames': {'$exists': True}}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

# Array fields and operators
criteria = {"prizes.category": "physics"}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

criteria = {"prizes.category": {"$ne": "physics"}}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

criteria = {"prizes.category": {"$in": ["physics", "chemistry", "medicine"]}}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

criteria ={"prizes.category": {"$nin": ["physics", "chemistry", "medicine"]}}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

# Enter $elemMatch
print('\nCount laureates who won unshared prizes in physics:')
criteria = {"prizes": {"category": "physics", "share": "1"}}
print(f'Incorrect --> {criteria}', db.laureates.count_documents(criteria)) # Structure not found

criteria = {"prizes.category": "physics", "prizes.share": "1"}
print(f'Incorrect --> {criteria}', db.laureates.count_documents(criteria)) # Match both not necessary in the same subdocument.

criteria = {"prizes": {"$elemMatch":{"category": "physics", "share": "1"}}}
print(criteria, db.laureates.count_documents(criteria)) # Perfect!

criteria = {"prizes": {"$elemMatch": {"category": "physics",
                                      "share": "1",
                                      "year": {"$lt": "1945"},}}}
print(f'\n{criteria}', db.laureates.count_documents(criteria))

print('*********************************************************')
print('** 02.09 Sharing in physics after World War II')
print('*********************************************************')
criteria = {"prizes": {"$elemMatch": {"category": "physics", 
                                      "share": "1", 
                                      "year": {"$gte": "1945"}}}}
unshared_prize = db.laureates.count_documents(criteria)
print('Unshared prize:', unshared_prize)

criteria = {"prizes": {"$elemMatch": {"category": "physics", 
                                      "share": {"$ne": "1"}, 
                                      "year": {"$gte": "1945"}}}}
shared_prize = db.laureates.count_documents(criteria)
print('Shared prize:', shared_prize)

print('Ratio:', unshared_prize/shared_prize)

print('*********************************************************')
print('** 02.10 Meanwhile, in other categories...')
print('*********************************************************')
# Save a filter for laureates with unshared prizes
unshared = {
    "prizes": {'$elemMatch': {
        'category': {'$nin': ["physics", "chemistry", "medicine"]},
        "share": "1",
        "year": {'$gte': "1945"},
    }}}

# Save a filter for laureates with shared prizes
shared = {
    "prizes": {'$elemMatch': {
        'category': {'$nin': ["physics", "chemistry", "medicine"]},
        "share": {'$ne': "1"},
        "year": {'$gte': "1945"},
    }}}

ratio = db.laureates.count_documents(unshared) / db.laureates.count_documents(shared)
print(ratio)

print('*********************************************************')
print('** 02.11 Organizations and prizes over time')
print('*********************************************************')
# Save a filter for organization laureates with prizes won before 1945
before = {
    'gender': 'org',
    'prizes.year': {'$lt': "1945"},
    }

# Save a filter for organization laureates with prizes won in or after 1945
in_or_after = {
    'gender': 'org',
    'prizes.year': {'$gte': "1945"},
    }

n_before = db.laureates.count_documents(before)
n_in_or_after = db.laureates.count_documents(in_or_after)
ratio = n_in_or_after / (n_in_or_after + n_before)
print(ratio)

print('*********************************************************')
print('** 02.12 Distinct As You Like It')
print('*********************************************************')
# Exploring
criteria = {"firstname": "Marie"}
print(f'Filter: {criteria} \nFound elements:')
pprint(db.laureates.find_one(criteria))

# Finding a substring with $regex
criteria = {"bornCountry": {"$regex": "Poland"}}
print(f'\nFilter: {criteria} \nFound elements:')
pprint(db.laureates.distinct('bornCountry', criteria))

# Flag options for regular expressions - using $regex
case_sensitive = db.laureates.distinct("bornCountry", {"bornCountry": {"$regex": "Poland"}})
pprint(case_sensitive)

case_insensitive = db.laureates.distinct("bornCountry", {"bornCountry": {"$regex": "poland", "$options": "i"}})
pprint(case_insensitive)

assert set(case_sensitive) == set(case_insensitive)

# Flag options for regular expressions - using bson.regex (the best option)
bson_option = db.laureates.distinct("bornCountry", {"bornCountry": Regex("poland", "i")})

assert set(case_sensitive) == set(bson_option)

# Flag options for regular expressions - using re (not recomended)
re_option = db.laureates.distinct("bornCountry", {"bornCountry": re.compile("poland", re.I)})

assert set(case_sensitive) == set(re_option)

# Beginning and ending (and escaping)
print('\nBegin with "Poland":')
pprint(db.laureates.distinct("bornCountry", {"bornCountry": Regex("^Poland")}))

print('\nBegin with "Poland (now"')
pprint(db.laureates.distinct("bornCountry", {"bornCountry": Regex("^Poland \(now")}))

print('\nEnd with "Poland)"')
pprint(db.laureates.distinct("bornCountry", {"bornCountry": Regex("now Poland\)$")}))

print('*********************************************************')
print('** 02.13 Glenn, George, and others in the G.B. crew')
print('*********************************************************')
print('Laureates that have first name beginning with "G" and last name begining with "S": ',
      db.laureates.count_documents({"firstname": Regex('^G'), "surname": Regex('^S')}))

print('*********************************************************')
print('** 02.14 Germany, then and now')
print('*********************************************************')
# Filter for laureates with "Germany" in their "bornCountry" value
criteria = {"bornCountry": Regex('Germany')}
print(f'Filter: {criteria} \nFound born country:')
pprint(set(db.laureates.distinct("bornCountry", criteria)))

# Filter for laureates with a "bornCountry" value starting with "Germany"
criteria = {"bornCountry": Regex('^Germany')}
print(f'\nFilter: {criteria} \nFound born country:')
pprint(set(db.laureates.distinct("bornCountry", criteria)))

# Fill in a string value to be sandwiched between the strings "^Germany " and "now"
criteria = {"bornCountry": Regex("^Germany " + '\(' + "now")}
print(f'\nFilter: {criteria} \nFound born country:')
pprint(set(db.laureates.distinct("bornCountry", criteria)))

#Filter for currently-Germany countries of birth. Fill in a string value to be sandwiched between the strings "now" and "$"
criteria = {"bornCountry": Regex("now" + ' Germany\)' + "$")}
print(f'\nFilter: {criteria} \nFound born country:')
pprint(set(db.laureates.distinct("bornCountry", criteria)))

print('*********************************************************')
print('** 02.15 The prized transistor')
print('*********************************************************')
# Save a filter for laureates with prize motivation values containing "transistor" as a substring
criteria = {'prizes.motivation': Regex('transistor')}

# Save the field names corresponding to a laureate's first name and last name
first, last = 'firstname', 'surname'
print([(laureate[first], laureate[last]) for laureate in db.laureates.find(criteria)])

print('*********************************************************')
print('END')
print('*********************************************************')