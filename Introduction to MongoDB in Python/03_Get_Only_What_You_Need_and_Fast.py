# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:25:40 2021

@author: jaces
"""
# Impor libraries
from pymongo import MongoClient
from pprint import pprint
from operator import itemgetter # Used in a sort options
from time import time
from collections import Counter

# Client connects to "localhost" by default
client = MongoClient()

# Connect to "nobel" database on the fly
db = client["nobel"]

print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 03. Get Only What You Need, and Fast')
print('*********************************************************')
print('** 03.01 Projection')
print('*********************************************************')
# Projection in MongoDB
print('** Projection in MongoDB...')
# Include only prizes.affiliations, exclude _id
docs = db.laureates.find(filter={}, 
                         projection={"prizes.affiliations": 1, "_id": 0})
print(type(docs))
docs = list(docs)

# size of docs
print('Size:', len(docs))

# convert to list and slice
pprint(docs[:3])

# Missing fields
print('\n** Missing fields')
# use "gender":"org" to select organizations organizations have no bornCountry
docs = db.laureates.find(filter={"gender": "org"},
                         projection=["bornCountry", "firstname"])
docs = list(docs)
print(len(docs))
pprint(docs[:2])

# only projected fields that exist are returned
docs = db.laureates.find({}, 
                         ["favoriteIceCreamFlavor"]) 
docs = list(docs)
print(len(docs))
pprint(docs[:2])

# Simple aggregation
print('\n** Simple aggregation')
docs = list(db.laureates.find({}, ["prizes"]))
n_prizes = 0
for doc in docs:
    # count the number of pizes in each doc
    n_prizes += len(doc["prizes"])
print(n_prizes)

# using comprehension
print(sum([len(doc["prizes"]) for doc in docs]))

print('*********************************************************')
print('** 03.02 Shares of the 1903 Prize in Physics')
print('*********************************************************')
criteria = {"prizes": {"$elemMatch": {"category": "physics", "year": "1903"}}}
fields_to_retrieve = {"firstname": 1, "surname": 1, "prizes.share": 1, "_id": 0}

pprint(db.laureates.find_one(criteria))
docs = list(db.laureates.find(filter = criteria, projection = fields_to_retrieve))

print(len(docs))
pprint(docs)

print('*********************************************************')
print('** 03.03 Rounding up the G.S. crew')
print('*********************************************************')
# Find laureates whose first name starts with "G" and last name starts with "S"
criteria = {"firstname": {"$regex": "^G"}, "surname": {"$regex": "^S"}}
docs = list(db.laureates.find(filter = criteria))
print(f'\nFilter: {criteria} \nFound docs: {len(docs)} \nFirst doc:')
pprint(docs[0])

# Use projection to select only firstname and surname
fields_to_retrieve = {"firstname": 1, "surname": 1, "_id": 0}
docs = list(db.laureates.find(filter = criteria, projection = fields_to_retrieve))
print(f'\nFilter: {criteria} \nFound docs: {len(docs)} \nFirst doc:')
pprint(docs[0])

# Iterate over docs and concatenate first name and surname
full_names = [doc["firstname"] + " " + doc["surname"]  for doc in docs]

# Print the full names
print('\nFinal result:')
pprint(full_names)

print('*********************************************************')
print('** 03.04 Doing our share of data validation')
print('*********************************************************')
# Save documents, projecting out laureates share
criteria = {}
fields_to_retrieve = ['laureates.share']
prizes = list(db.prizes.find(criteria, fields_to_retrieve))

print(f'Prizes found: {len(prizes)} \nFirst prize found:')
pprint(prizes[0])

total_share_not_one = []
# Iterate over prizes
for prize in prizes:
    # Initialize total share
    total_share = 0
    
    # Iterate over laureates for the prize
    for laureate in prize["laureates"]:
        # add the share of the laureate to total_share
        total_share += 1 / float(laureate['share'])
        
    # Print the total share if not one    
    if total_share != 1: 
        print(total_share)    
        total_share_not_one.append(laureate['_id'])

if len(total_share_not_one) == 0:
    print('All share prizess add up to 1!')
    
print('*********************************************************')
print('** 03.05 Sorting')
print('*********************************************************')
print('The data...')
# Just getting the data
docs = list(db.prizes.find({"category": "physics"}, ["year"]))
print([doc["year"] for doc in docs][:5])


print('\nSorting data with Python...')
# Sorting post-query with itemgetter, ascending
docs = sorted(docs, key=itemgetter("year"))
print([doc["year"] for doc in docs][:5])

# Sorting post-query with itemgetter, descending
docs = sorted(docs, key=itemgetter("year"), reverse=True)
print([doc["year"] for doc in docs][:5])

print('\nSorting data with MongoDB...')
# Sorting in-query with MongoDB - ascending
cursor = db.prizes.find({"category": "physics"}, ["year"], sort=[("year", 1)])
print([doc["year"] for doc in cursor][:5])

# Sorting in-query with MongoDB - descending
cursor = db.prizes.find({"category": "physics"}, ["year"],
sort=[("year", -1)])
print([doc["year"] for doc in cursor][:5])

print('\nSorting data with MongoDB (base on 2 fields)...')
# Primary and secondary sorting
for doc in db.prizes.find(filter     = {"year": {"$gt": "1966", "$lt": "1970"}}, 
                          projection = ["category", "year"], 
                          sort       = [("year", 1), ("category", -1)]):
    print("{year} {category}".format(**doc))

print('*********************************************************')
print('** 03.06 What the sort?')
print('*********************************************************')
docs = list(db.laureates.find(
    {"born": {"$gte": "1900"}, "prizes.year": {"$gte": "1954"}},
    {"born": 1, "prizes.year": 1, "_id": 0},
    sort=[('prizes.year', 1), ('born', -1)]))

for doc in docs[:5]:
    print(doc)

print('*********************************************************')
print('** 03.07 Sorting together: MongoDB + Python')
print('*********************************************************')
# Definition of all_laureates function
def all_laureates(prize):  
    """Sort the laureates by surname"""
    sorted_laureates = sorted(prize['laureates'], key=itemgetter('surname'))
    
    # extract surnames
    surnames = [laureate['surname'] for laureate in sorted_laureates]
    
    # concatenate surnames separated with " and " 
    all_names = " and ".join(surnames)
    
    return all_names

# Finding one document to remember her structure.
sample_prize = db.prizes.find_one({})
pprint(sample_prize)

# test the function on a sample doc
print(all_laureates(sample_prize))

# find physics prizes, project year and first and last name, and sort by year
docs = db.prizes.find(filter= {'category': 'physics'}, 
                      projection= ["year", "laureates.firstname", "laureates.surname"], 
                      sort= [('year', 1)])
docs = list(docs)
pprint(docs[0])

# print the year and laureate names (from all_laureates)
for doc in docs:
    print("{year}: {names}".format(year=doc['year'], names=all_laureates(doc)))

print('*********************************************************')
print('** 03.08 Gap years')
print('*********************************************************')
# original categories from 1901
original_categories = db.prizes.distinct('category', {'year': '1901'})
print(original_categories)

# project year and category, and sort
docs = db.prizes.find(
        filter={},
        projection = {'year': 1, 'category': 1, '_id': 0},
        sort = [('year', -1), ('category', 1)]
)

#print the documents
for doc in docs:
        print(doc)
        
print('*********************************************************')
print('** 03.09 What are indexes?')
print('*********************************************************')
if 'year_1' in db.prizes.index_information():
    db.prizes.drop_index('year_1')
    
if 'category_year_1' in db.prizes.index_information():
    db.prizes.drop_index('category_year_1')

if 'firstname_bornCountry' in db.laureates.index_information():
    db.laureates.drop_index('firstname_bornCountry')



print('** BEFORE SINGLE INDEX...')
start = time()
docs = list(db.prizes.find({"year": "1901"}))
print('find({"year": "1901"}):', time() - start, 'ms.')

start = time()
docs = list(db.prizes.find({}, sort=[("year", 1)]))
print('.find({}, sort=[("year", 1)]):', time() - start, 'ms.')



print('\n** AFTER SINGLE INDEX...')
# Adding a single-field index
_ = db.prizes.create_index([("year", 1)], name='year_1')

# Previously: 524 μs ± 7.34 μs
start = time()
docs = list(db.prizes.find({"year": "1901"}))
print('.find({"year": "1901"}):', time() - start, 'ms.')

# Previously: 5.18 ms ± 54.9 μs
start = time()
docs = list(db.prizes.find({}, sort=[("year", 1)]))
print('.find({}, sort=[("year", 1)]):', time() - start, 'ms.')

db.prizes.drop_index('year_1')



print('\n** BEFORE COMPOUND INDEX...')
start = time()
_ = list(db.prizes.find({"category": "economics"}, {"year": 1, "_id": 0}))
print('.find({"category": "economics"}, {"year": 1, "_id": 0}):', time() - start, 'ms.')

start = time()
_ = db.prizes.find_one({"category": "economics"}, {"year": 1, "_id": 0}, sort=[("year", 1)])
print('.find_one({"category": "economics"}, {"year": 1, "_id": 0}, sort=[("year", 1)]):', time() - start, 'ms.')



print('\n** AFTER COMPOUD INDEX...')
# Adding a compound (multiple-field) index
_ = db.prizes.create_index([("category", 1), ("year", 1)], name='category_year_1')

start = time()
_ = list(db.prizes.find({"category": "economics"}, {"year": 1, "_id": 0}))
print('.find({"category": "economics"}, {"year": 1, "_id": 0}):', time() - start, 'ms.')

start = time()
db.prizes.find_one({"category": "economics"}, {"year": 1, "_id": 0}, sort=[("year", 1)])
print('.find_one({"category": "economics"}, {"year": 1, "_id": 0}, sort=[("year", 1)]):', time() - start, 'ms.')

db.prizes.drop_index('category_year_1')



print('\n** INFORMATION ABOUT INDEX:')
print('Existing indexes in the "Laureates" collextion:')
pprint(db.laureates.index_information())

print('Process used by MongoDB before creation of index:')
pprint(db.laureates.find({"firstname": "Marie"}, {"bornCountry": 1, "_id": 0}).explain())

print('Process used by MongoDB after creation of index:')
_ = db.laureates.create_index([("firstname", 1), ("bornCountry", 1)], name='firstname_bornCountry')
pprint(db.laureates.find({"firstname": "Marie"}, {"bornCountry": 1, "_id": 0}).explain())
db.laureates.drop_index('firstname_bornCountry')

print('*********************************************************')
print('** 03.10 High-share categories')
print('*********************************************************')
# Specify an index model for compound sorting
index_model = [('category', 1), ('year', -1)]
db.prizes.create_index(index_model)

# Collect the last single-laureate year for each category
report = ""
for category in sorted(db.prizes.distinct("category")):
    doc = db.prizes.find_one(
        {'category': category, "laureates.share": "1"},
        sort=[('year', -1)]
    )
    report += "{category}: {year}\n".format(**doc)

print(report)

print('*********************************************************')
print('** 03.11 Recently single?')
print('*********************************************************')
# Specify an index model for compound sorting
index_model = [('category', 1), ('year', -1)]
db.prizes.create_index(index_model)

# Collect the last single-laureate year for each category
report = ""
for category in sorted(db.prizes.distinct("category")):
    doc = db.prizes.find_one(
        {'category': category, "laureates.share": "1"},
        sort=[('year', -1)]
    )
    report += "{category}: {year}\n".format(**doc)

print(report)

print('*********************************************************')
print('** 03.12 Born and affiliated')
print('*********************************************************')
# Ensure an index on country of birth
db.laureates.create_index([('bornCountry', 1)])

# Collect a count of laureates for each country of birth
n_born_and_affiliated = {
    country: db.laureates.count_documents({
        'bornCountry': country,
        "prizes.affiliations.country": country
    })
    for country in db.laureates.distinct("bornCountry")
}

five_most_common = Counter(n_born_and_affiliated).most_common(5)
pprint(five_most_common)

print('*********************************************************')
print('** 03.13 Limits')
print('*********************************************************')
# Exploring prizes collections...
print('** Exploring prizes collections...')
pprint(db.prizes.find_one({}))


# Limiting our exploration
print('\n** Limiting our exploration...')
for doc in db.prizes.find({}, ["laureates.share"]):
    share_is_three = [laureate["share"] == "3" for laureate in doc["laureates"]]
    
    assert all(share_is_three) or not any(share_is_three)
    
for doc in db.prizes.find({"laureates.share": "3"}, limit=6):
    print("{year} {category}".format(**doc))


# Skips and paging through results
print('\n** Skips and paging through results...')
for doc in db.prizes.find({"laureates.share": "3"}, skip=2, limit=2):
    print("{year} {category}".format(**doc))
for doc in db.prizes.find({"laureates.share": "3"}, skip=4, limit=2):
    print("{year} {category}".format(**doc))


# Using cursor methods for {sort, skip, limit}
print('\n** Using cursor methods for {sort, skip, limit}...')
for doc in db.prizes.find({"laureates.share": "3"}).limit(3):
    print("{year} {category}".format(**doc))
    
for doc in (db.prizes.find({"laureates.share": "3"}).skip(3).limit(3)):
    print("{year} {category}".format(**doc))

for doc in (db.prizes.find({"laureates.share": "3"}).sort([("year", 1)]).skip(3).limit(3)):
    print("{year} {category}".format(**doc))


# Simpler sorts of sort
print('\n** Simpler sorts of sort...')
cursor1 = (db.prizes.find({"laureates.share": "3"}).skip(3).limit(3).sort([("year", 1)]))
cursor2 = (db.prizes.find({"laureates.share": "3"}).skip(3).limit(3).sort("year", 1))
cursor3 = (db.prizes.find({"laureates.share": "3"}).skip(3).limit(3).sort("year"))

docs = list(cursor1)
assert docs == list(cursor2) == list(cursor3)

for doc in docs:
    print("{year} {category}".format(**doc))

doc = db.prizes.find_one({"laureates.share": "3"}, skip=3, sort=[("year", 1)])
print("{year} {category}".format(**doc))

print('*********************************************************')
print('** 03.14 Setting a new limit?')
print('*********************************************************')
pprint(list(db.prizes.find({"category": "economics"}, {"year": 1, "_id": 0}).sort("year").limit(3).limit(5)))

print('*********************************************************')
print('** 03.15 The first five prizes with quarter shares')
print('*********************************************************')
# Fetch prizes with quarter-share laureate(s)
filter_ = {'laureates.share': '4'}

# Save the list of field names
projection = ['category', 'year', 'laureates.motivation']

# Save a cursor to yield the first five prizes
cursor = db.prizes.find(filter_, projection).sort('year').limit(5)
pprint(list(cursor))

print('*********************************************************')
print('** 03.16 Pages of particle-prized people')
print('*********************************************************')
# Explore the strycture
pprint(db.laureates.find_one({}))

# Write a function to retrieve a page of data
def get_particle_laureates(page_number=1, page_size=3):
    if page_number < 1 or not isinstance(page_number, int):
        raise ValueError("Pages are natural numbers (starting from 1).")
    particle_laureates = list(
        db.laureates.find(
            {'prizes.motivation': {'$regex': "particle"}},
            ["firstname", "surname", "prizes"])
        .sort([('prizes.year', 1), ('surname', 1)])
        .skip(page_size * (page_number - 1))
        .limit(page_size))
    return particle_laureates

# Collect and save the first nine pages
pages = [get_particle_laureates(page_number=page) for page in range(1,9)]
pprint(pages[0])

print('*********************************************************')
print('END')
print('*********************************************************')