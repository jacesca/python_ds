# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:34:00 2021

@author: jaces
"""
# Import libraries
from pymongo import MongoClient
from pprint import pprint
from collections import OrderedDict
from itertools import groupby
from operator import itemgetter # Used in a sort options

# Client connects to "localhost" by default
client = MongoClient()

# Connect to "nobel" database on the fly
db = client["nobel"]


print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 04. Aggregation Pipelines: Let the Server Do It For You')
print('*********************************************************')
print('** 04.01 Intro to Aggregation')
print('*********************************************************')
# Exploring the data
print('** One document from "laureates" collection...')
pprint(db.laureates.find_one({}))

# Queries have implicit stages
print('\n** Method: Find \nFilter: {"bornCountry": "USA"}, limit=3')
cursor = db.laureates.find(
            filter={"bornCountry": "USA"},
            projection={"prizes.year": 1},
            limit=3
         )
for doc in cursor:
    print(doc["prizes"])
    
# Queries have implicit stages - aggregation
print('\n** Method: Aggregate \nMatch: {"bornCountry": "USA"}, limit=3')
cursor = db.laureates.aggregate([
            {"$match": {"bornCountry": "USA"}},
            {"$project": {"prizes.year": 1}},
            {"$limit": 3}
         ])
for doc in cursor:
    print(doc["prizes"])

# Adding sort and skip stages
print('\n** Method: Aggregate - Sort')
cursor = list(db.laureates.aggregate([
                {"$match": {"bornCountry": "USA"}},
                {"$project": {"prizes.year": 1, "_id": 0}},
                {"$sort": OrderedDict([("prizes.year", 1)])},
                {"$skip": 1},
                {"$limit": 3}
         ]))
pprint(cursor)

# But can I count? - aggregation
print('\n** Method: Aggregate - Count')
cursor = list(db.laureates.aggregate([
                {"$match": {"bornCountry": "USA"}},
                {"$count": "n_USA-born-laureates"}
         ]))
print(cursor)

# But can I count? - count_documents
print('\n** Method: count_documents')
print(db.laureates.count_documents({"bornCountry": "USA"}))
    
print('*********************************************************')
print('** 04.02 Sequencing stages')
print('*********************************************************')
cursor = (db.laureates.find(
    projection={"firstname": 1, "prizes.year": 1, "_id": 0},
    filter={"gender": "org"})
 .limit(3).sort("prizes.year", -1))

project_stage = {"$project": {"firstname": 1, "prizes.year": 1, "_id": 0}}
match_stage = {"$match": {"gender": "org"}}
limit_stage = {"$limit": 3}
sort_stage = {"$sort": {"prizes.year": -1}}

print('Using db.laureates.find...')
pprint(list(cursor))

cursor = list(db.laureates.aggregate([match_stage, project_stage, sort_stage, limit_stage]))
print('\nUsing db.laureates.aggregate...')
pprint(cursor)

print('*********************************************************')
print("** 04.03 Aggregating a few individuals' country data")
print('*********************************************************')
print('Using find method...')
cursor = (db.laureates.find(
    {"gender": {"$ne": "org"}},
    ["bornCountry", "prizes.affiliations.country"]
).limit(3))
pprint(list(cursor))

# Translate cursor to aggregation pipeline
print('\nUsing find aggregate...')
pipeline = [
    {"$match": {"gender": {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$limit": 3}
]

for doc in db.laureates.aggregate(pipeline):
    print("{bornCountry}: {prizes}".format(**doc))
    

print('*********************************************************')
print('** 04.04 Passing the aggregation baton to Python')
print('*********************************************************')
# Exploring the data
print('One document from "prizes" collection...')
pprint(db.prizes.find_one({}))

original_categories = set(db.prizes.distinct("category", {"year": "1901"}))

# Save an pipeline to collect original-category prizes
pipeline = [
    {'$match': {'category': {'$in': list(original_categories)}}},
    {'$project': {'year': 1, 'category': 1}},
    {'$sort': OrderedDict([('year', -1)])}
]

cursor = db.prizes.aggregate(pipeline)
for key, group in groupby(cursor, key=itemgetter("year")):
    missing = original_categories - {doc["category"] for doc in group}
    if missing:
        print("{year}: {missing}".format(year=key, missing=", ".join(sorted(missing))))
        
print('*********************************************************')
print('** 04.05 Aggregation Operators and Grouping')
print('*********************************************************')
# Field paths
print('FIELDS PATH...')
print(list(db.laureates.aggregate([{"$project": {"prizes.share": 1, '_id': 0}}]))[:2])
print(db.laureates.aggregate([{"$project": {"prizes.share": 1, '_id': 0}}]).next())

print(db.laureates.aggregate([{"$project": {"n_prizes": {"$size": "$prizes"}, '_id': 0}}]).next())

# Operator expressions
# We could also write the operator expression as taking a list of one element, and we get the same result.
# For convenience, when an operator only has one parameter, we can omit the brackets as above.
print('\nOPERATOR EXPRESSIONS...')
print(db.laureates.aggregate([{"$project": {"n_prizes": {"$size": ["$prizes"]}, '_id': 0}}]).next())

# One more example: a multi-parameter operator
# Here I use the dollar-in operator, which takes two parameters. I then project a new field, "solo winner", 
# which is true if and only if the array of prize shares contains the string value "1".
print('\nMULTI PARAMETER OPERATOR EXPRESSIONS...')
print(db.laureates.aggregate([{"$project": {"solo_winner": {"$in": ["1", "$prizes.share"]}, '_id': 0}}]).next())

# Implementing .distinct()
list_1 = list(db.laureates.distinct("bornCountry", 
                                    {"prizes.share": "4"}))
pprint(list_1)
# A group stage takes an expression object that must map the underscore-id field. 
# In this case, each output document will have as its id a distinct value of the bornCountry field.
# This includes the value None, which happens when a field is not present in a document.
list_2 = list(db.laureates.aggregate([
            {"$match": {"prizes.share": "4"}},
            {"$group": {"_id": "$bornCountry"}},
         ]))
pprint(list_2)

list_2 = [doc["_id"] for doc in list_2]
print(set(list_2) - {None} == set(list_1))

# How many prizes have been awarded in total?
print("\nHow many prizes have been awarded in total?")
data = list(db.laureates.aggregate([
                {"$match"  : {"prizes.share": "4"}},
                {"$project": {"n_prizes": {"$size": "$prizes"}}},
                {"$group": {"_id": None, "n_prizes_total": {"$sum": "$n_prizes"}}}
        ]))
print(data)
# How many prizes have been awarded per born country?
print("\nHow many prizes have been awarded per born country?")
data = list(db.laureates.aggregate([
                {"$match"  : {"prizes.share": "4"}},
                {"$project": {'bornCountry': 1, 
                              "n_prizes": {"$size": "$prizes"}}},
                {"$group"  : {"_id": '$bornCountry', 
                              "n_prizes_total": {"$sum": "$n_prizes"}}}
        ]))
pprint(data)

print('*********************************************************')
print('** 04.06 Field Paths and Sets')
print('*********************************************************')
# Limiting our exploration
for doc in db.prizes.find({}, ["laureates.share"]):
    share_is_three = [laureate["share"] == "3" for laureate in doc["laureates"]]
    
    assert all(share_is_three) or not any(share_is_three)
print(share_is_three)

print(list(db.prizes.aggregate([
        {"$project": {"allThree": {"$setEquals": [[3], '$laureates.share']},
                      "noneThree": {"$not": {"$setIsSubset": [[3], '$laureates.share']}}}},
        {"$match": {"$nor": [{"allThree": True}, {"noneThree": True}]}}])))

print('*********************************************************')
print('** 04.07 Organizing prizes')
print('*********************************************************')
# Count prizes awarded (at least partly) to organizations as a sum over sizes of "prizes" arrays.
pipeline = [
    {'$match': {'gender': "org"}},
    {"$project": {"n_prizes": {"$size": "$prizes"}}},
    {"$group": {"_id": None, "n_prizes_total": {"$sum": '$n_prizes'}}}
]

print(list(db.laureates.aggregate(pipeline)))

print('*********************************************************')
print('** 04.08 Gap years, aggregated')
print('*********************************************************')
original_categories = sorted(set(db.prizes.distinct("category", {"year": "1901"})))
pipeline = [
    {"$match": {"category": {"$in": original_categories}}},
    {"$project": {"category": 1, "year": 1}},
    
    # Collect the set of category values for each prize year.
    {"$group": {"_id": '$year', "categories": {"$addToSet": "$category"}}},
    
    # Project categories *not* awarded (i.e., that are missing this year).
    {"$project": {"missing": {"$setDifference": [original_categories, '$categories']}}},
    
    # Only include years with at least one missing category
    {"$match": {"missing.0": {"$exists": True}}},
    
    # Sort in reverse chronological order. Note that "_id" is a distinct year at this stage.
    {"$sort": OrderedDict([("_id", -1)])},
]

for doc in db.prizes.aggregate(pipeline):
    print("{year}: {missing}".format(year=doc["_id"],missing=", ".join(sorted(doc["missing"]))))
    
print('*********************************************************')
print('** 04.09 Zoom into Array Fields')
print('*********************************************************')
# Sizing and summing
print('** SIZING AND SUMMING...')
print('Laureates per year/category before 1903:')
data = list(db.prizes.aggregate([
                {"$project": {"year"       : 1, 
                              "category"   : 1, 
                              "n_laureates": {"$size": "$laureates"},
                              "_id"        : 0}},
                {"$match"  : {'year'       : {"$lt"  : '1903'}}},
                {"$sort"   : {'category'   : 1}}
       ]))
pprint(data)

print('\nLaureates per category before 1903:')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'       : {"$lt"  : '1903'}}},
                {"$project": {"category"   : 1,
                              "n_laureates": {"$size": "$laureates"}}},
                {"$group"  : {"_id"        : "$category", 
                              "n_laureates": {"$sum" : "$n_laureates"}}},
                {"$sort"   : {"n_laureates": -1}},
       ]))
pprint(data)

# How to $unwind
print('\n** HOW TO $UNWIND...')
print('Laureatees in 1901 present in prizes collections:')
data = list(db.prizes.aggregate([
                {"$match"  : {'year': '1901'}},
                {"$unwind" : "$laureates"},
                {"$project": {"_id": 0, 
                              "year": 1, 
                              "category": 1,
                              "laureates.id": 1,
                              "laureates.surname": 1, 
                              "laureates.share": 1}},
                {"$sort"   : {'category'   : 1}}
               #{"$limit"   : 3}
       ]))
pprint(data)

# Renormalization, anyone?
print('\n** RENORMALIZATION, ANYONE?...')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'        : '1901'}},
                {"$unwind" : "$laureates"},
                {"$project": {"year"        : 1, 
                              "category"    : 1, 
                              "laureates.id": 1}},
                {"$group"  : {"_id"         : {"$concat": ["$category", ":", "$year"]},
                              "laureate_ids": {"$addToSet": "$laureates.id"}}},
                {"$sort"   : {'_id'   : 1}}
               #{"$limit": 5}
       ]))
pprint(data)

# $unwind and count 'em, one by one
print("\n$UNWIND AND COUNT 'EM, ONE BY ONE...")
print('Laureates per category in 1901 (using ""$group"):')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'       : '1901'}},
                {"$project": {"category"   : 1,
                              "n_laureates": {"$size": "$laureates"}}},
                {"$group"  : {"_id"        : "$category", 
                              "n_laureates": {"$sum" : "$n_laureates"}}},
                {"$sort"   : {"n_laureates": -1}},
       ]))
pprint(data)

print('\nLaureates per category in 1901 (using ""$unwind"):')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'       : '1901'}},
                {"$unwind" : "$laureates"},
                {"$group"  : {"_id"        : "$category", 
                              "n_laureates": {"$sum" : 1}}},
                {"$sort"   : {"n_laureates": -1}},
       ]))
pprint(data)

# $lookup
print('\n$LOOKUP')
# This stage pulls in documents from another collection via what's termed a left outer join. 
# Let's collect countries of birth for economics laureates.
print('Laureates per category in 1901:')
data = list(db.prizes.aggregate([
                {"$match": {'year'       : '1901'}},
                {"$unwind": "$laureates"},
                {"$lookup": {"from": "laureates", "foreignField": "id",
                             "localField": "laureates.id", "as": "laureate_bios"}},
                {"$project": {"category": 1,
                              'laureate_bios.id': 1,
                              'laureate_bios.surname':1,
                              "laureate_bios.bornCountry": 1,
                              '_id': 0}},
                {"$sort"   : {"category": 1}},
       ]))
pprint(data)

print('\nReconfigurating to group the bornCountries per category in 1901:')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'       : '1901'}},
                {"$unwind" : "$laureates"},
                {"$lookup" : {"from": "laureates", "foreignField": "id",
                             "localField": "laureates.id", "as": "laureate_bios"}},
                {"$project": {"category": 1,
                              'laureate_bios.id': 1,
                              "laureate_bios.bornCountry": 1,
                              '_id': 0}},
                {"$unwind" : "$laureate_bios"},
                {"$group"  : {"_id": '$category',
                              "bornCountries": {"$addToSet": "$laureate_bios.bornCountry"}}},
                {"$sort"   : {"_id": 1, 
                              'bornCountries': 1}},
       ]))
pprint(data)

print('\nReconfigurating to group the bornCountries in 1901:')
data = list(db.prizes.aggregate([
                {"$match"  : {'year'       : '1901'}},
                {"$unwind" : "$laureates"},
                {"$lookup" : {"from": "laureates", "foreignField": "id",
                             "localField": "laureates.id", "as": "laureate_bios"}},
                {"$project": {"category": 1,
                              'laureate_bios.id': 1,
                              "laureate_bios.bornCountry": 1,
                              '_id': 0}},
                {"$unwind" : "$laureate_bios"},
                {"$group"  : {"_id": None,
                              "bornCountries": {"$addToSet": "$laureate_bios.bornCountry"}}},
                {"$sort"   : {"_id": 1, 
                              'bornCountries': 1}},
       ]))
pprint(data)
print(data[0]['bornCountries'])

print('\nTaking the data from laureates collections:')
bornCountries = db.laureates.distinct(
                    key = "bornCountry", 
                    filter = {"prizes.year": "1901"}
                )
print(bornCountries)
assert set(bornCountries) == set(data[0]['bornCountries'])

print('*********************************************************')
print('** 04.10 Embedding aggregation expressions')
print('*********************************************************')
#print('Exploring laureates collection...')
#pprint(db.laureates.find_one({'bornCountry': {'$exists': False}}))
#pprint(db.laureates.find_one({'prizes.year': {'$exists': False}}))

#print('\nDistinct bornCountry per year...')
#data = list(db.laureates.aggregate([
#                {"$unwind" : "$prizes"},
#                {"$project": {"prizes.year" : 1, 
#                              "bornCountry" : 1}},
#                {"$group"  : {"_id"         : '$prizes.year',
#                              "bornCountries": {"$addToSet": "$bornCountry"}}},
#                {"$sort"   : {'_id'   : 1}}
#       ]))
#pprint(data)

data = db.laureates.distinct("bornCountry", {'prizes.year': '1944'})
print('\nDistinct bornCountry in 1944:', data)

data = db.laureates.count_documents({'bornCountry': {'$exists': False}, 'prizes.year': '1944'})
print('Documents with no bornCountry in 1944: ', data)

assert all(isinstance(v, str) for v in set(db.laureates.distinct("bornCountry")) - {None})

print('\nDocuments found in 1944 with bornCountry:')
data = db.laureates.count_documents({'bornCountry': {'$exists': True}, 
                                     'prizes.year': '1944'})
print('Result (1st option): ', data)
data = db.laureates.count_documents({"bornCountry": {"$in": db.laureates.distinct("bornCountry", {'prizes.year': '1944'})}, 
                                     'prizes.year': '1944'})
print('Result (2nd option): ', data)
data = db.laureates.count_documents({"$expr": {"$in": ["$bornCountry", db.laureates.distinct("bornCountry", 
                                                                                             {'prizes.year': '1944'})]}, 
                                     'prizes.year': '1944'})
print('Result (3rd option): ', data)
data = db.laureates.count_documents({"$expr": {"$eq": [{"$type": "$bornCountry"}, "string"]}, 
                                     'prizes.year': '1944'})
print('Result (4th option): ', data)
data = db.laureates.count_documents({"bornCountry": {"$type": "string"}, 
                                     'prizes.year': '1944'})
print('Result (5th option): ', data)

print('*********************************************************')
print('** 04.11 Here and elsewhere')
print('*********************************************************')
print('1st stage...')
pipeline = [{"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # For exploration purpose
            {'$limit': 3}]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n2nd stage...')
pipeline = [{"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            {'$unwind': "$prizes.affiliations"},
            # For exploration purpose
            {'$limit': 3}] 
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n3rd stage...')
pipeline = [{"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            {'$unwind': "$prizes.affiliations"},
            # Ensure values in the list of distinct values (so not empty)
            {"$match": {"prizes.affiliations.country": {'$in': db.laureates.distinct("prizes.affiliations.country")}}},
            # For exploration purpose
            {'$limit': 3}] 
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n4th stage...')
pipeline = [{"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            {'$unwind': "$prizes.affiliations"},
            # Ensure values in the list of distinct values (so not empty)
            {"$match": {"prizes.affiliations.country": {'$in': db.laureates.distinct("prizes.affiliations.country")}}},
            # Reproject the data
            {"$project": {"affilCountrySameAsBorn": {"$gte": [{"$indexOfBytes": ["$prizes.affiliations.country", 
                                                                                 "$bornCountry"]}, 0]}}},
            # For exploration purpose
            {'$limit': 3}] 
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n5th stage...')
pipeline = [{"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            {'$unwind': "$prizes.affiliations"},
            # Ensure values in the list of distinct values (so not empty)
            {"$match": {"prizes.affiliations.country": {'$in': db.laureates.distinct("prizes.affiliations.country")}}},
            # Reproject the data
            {"$project": {"affilCountrySameAsBorn": {"$gte": [{"$indexOfBytes": ["$prizes.affiliations.country", 
                                                                                 "$bornCountry"]}, 0]}}},
            # Count by "$affilCountrySameAsBorn" value (True or False)
            {"$group": {"_id": "$affilCountrySameAsBorn",
                        "count": {"$sum": 1}}},] 
data = list(db.laureates.aggregate(pipeline))
print(data)

print('\nAll process at once:')
key_ac = "prizes.affiliations.country"
key_bc = "bornCountry"
pipeline = [
    {"$project": {key_bc: 1, key_ac: 1}},

    # Ensure a single prize affiliation country per pipeline document
    {'$unwind': "$prizes"},
    {'$unwind': "$prizes.affiliations"},

    # Ensure values in the list of distinct values (so not empty)
    {"$match": {key_ac: {'$in': db.laureates.distinct(key_ac)}}},
    {"$project": {"affilCountrySameAsBorn": {
        "$gte": [{"$indexOfBytes": ["$"+key_ac, "$"+key_bc]}, 0]}}},

    # Count by "$affilCountrySameAsBorn" value (True or False)
    {"$group": {"_id": "$affilCountrySameAsBorn",
                "count": {"$sum": 1}}},
]
for doc in db.laureates.aggregate(pipeline): print(doc)

print('*********************************************************')
print('** 04.12 Countries of birth by prize category')
print('*********************************************************')
print('1st stage...')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    # For exploration purpose
    {'$limit': 1},
]
data = list(db.prizes.aggregate(pipeline))
pprint(data)

print('\n2nd stage...')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    # Link with laureates collection
    {"$lookup": {"from": "laureates", "foreignField": "id",
                 "localField": "laureates.id", "as": "laureate_bios"}},
    # Unwind the new laureate_bios array
    {"$unwind": '$laureate_bios'},
    # For exploration purpose
    {'$limit': 1},
]
data = list(db.prizes.aggregate(pipeline))
pprint(data)

print('\n3rd stage...')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    {"$lookup": {"from": "laureates", "foreignField": "id",
                 "localField": "laureates.id", "as": "laureate_bios"}},
    # Unwind the new laureate_bios array
    {"$unwind": '$laureate_bios'},
    {"$project": {"category": 1,
                  "bornCountry": "$laureate_bios.bornCountry",
                  '_id': 0}},
    # For exploration purpose
    {'$limit': 3},
]
data = list(db.prizes.aggregate(pipeline))
pprint(data)

print('\n4th stage...')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    {"$lookup": {"from": "laureates", "foreignField": "id",
                 "localField": "laureates.id", "as": "laureate_bios"}},
    # Unwind the new laureate_bios array
    {"$unwind": '$laureate_bios'},
    {"$project": {"category": 1,
                  "bornCountry": "$laureate_bios.bornCountry",
                  '_id': 0}},
    # Collect bornCountry values associated with each prize category
    {"$group": {'_id': "$category",
                "bornCountries": {"$addToSet": "$bornCountry"}}},
    # For exploration purpose
    {'$limit': 1},
]
data = list(db.prizes.aggregate(pipeline))
pprint(data)

print('\n5th stage...')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    {"$lookup": {"from": "laureates", "foreignField": "id",
                 "localField": "laureates.id", "as": "laureate_bios"}},
    # Unwind the new laureate_bios array
    {"$unwind": '$laureate_bios'},
    {"$project": {"category": 1,
                  "bornCountry": "$laureate_bios.bornCountry",
                  '_id': 0}},
    # Collect bornCountry values associated with each prize category
    {"$group": {'_id': "$category",
                "bornCountries": {"$addToSet": "$bornCountry"}}},
    # Project out the size of each category's (set of) bornCountries
    {"$project": {"category": 1,
                  "nBornCountries": {"$size": "$bornCountries"}}},
    {"$sort": {"nBornCountries": -1}},
]
data = list(db.prizes.aggregate(pipeline))
pprint(data)

print('\nAll process at once:')
pipeline = [
    # Unwind the laureates array
    {'$unwind': "$laureates"},
    {"$lookup": {
        "from": "laureates", "foreignField": "id",
        "localField": "laureates.id", "as": "laureate_bios"}},

    # Unwind the new laureate_bios array
    {"$unwind": '$laureate_bios'},
    {"$project": {"category": 1,
                  "bornCountry": "$laureate_bios.bornCountry"}},

    # Collect bornCountry values associated with each prize category
    {"$group": {'_id': "$category",
                "bornCountries": {"$addToSet": "$bornCountry"}}},

    # Project out the size of each category's (set of) bornCountries
    {"$project": {"category": 1,
                  "nBornCountries": {"$size": '$bornCountries'}}},
    {"$sort": {"nBornCountries": -1}},
]
for doc in db.prizes.aggregate(pipeline): print(doc)

print('*********************************************************')
print('** 04.13 Something Extra: $addFields to Aid Analysis')
print('*********************************************************')
print('1st stage...')
pipeline = [
    {"$project": {"died": {"$dateFromString": {"dateString": "$died"}},
                  "born": {"$dateFromString": {"dateString": "$born"}}}},
    # For exploration purpose
    {'$limit': 1}
]
try: 
    data = list(db.laureates.aggregate(pipeline))
    pprint(data)
except:
    print('Error found in dates!')

print('\n2nd stage...')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$project": {"died": {"$dateFromString": {"dateString": "$died"}},
                  "born": {"$dateFromString": {"dateString": "$born"}}}},
    # For exploration purpose
    {'$limit': 1}
]
try: 
    data = list(db.laureates.aggregate(pipeline))
    pprint(data)
except:
    print('Error found in dates!')
    
print('\n3rd stage...')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$addFields": {"bornArray": {"$split": ["$born", "-"]},
                    "diedArray": {"$split": ["$died", "-"]}}},
    # For exploration purpose
    {'$limit': 1}
]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n4th stage...')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$addFields": {"bornArray": {"$split": ["$born", "-"]},
                    "diedArray": {"$split": ["$died", "-"]}}},
    {"$addFields": {"born": {"$cond": [{"$in": ["00", "$bornArray"]},
                                       {"$concat": [{"$arrayElemAt": ["$bornArray", 0]}, "-01-01"]},
                                       "$born"]}}},
    # For exploration purpose
    {'$limit': 1}
]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n5th stage...')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$addFields": {"bornArray": {"$split": ["$born", "-"]},
                    "diedArray": {"$split": ["$died", "-"]}}},
    {"$addFields": {"born": {"$cond": [{"$in": ["00", "$bornArray"]},
                                       {"$concat": [{"$arrayElemAt": ["$bornArray", 0]}, "-01-01"]},
                                       "$born"]}}},
    {"$project": {"died": {"$dateFromString": {"dateString": "$died"}},
                  "born": {"$dateFromString": {"dateString": "$born"}},
                  "_id": 0}},
    {"$project": {"years": {"$floor": {"$divide": [{"$subtract": ["$died", "$born"]},
                                                   31557600000]}}}}, # 1000 * 60 * 60 * 24 * 365.25 ms
    # For exploration purpose
    {'$limit': 3}
]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('\n6th stage...')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$addFields": {"bornArray": {"$split": ["$born", "-"]},
                    "diedArray": {"$split": ["$died", "-"]}}},
    {"$addFields": {"born": {"$cond": [{"$in": ["00", "$bornArray"]},
                                       {"$concat": [{"$arrayElemAt": ["$bornArray", 0]}, "-01-01"]},
                                       "$born"]}}},
    {"$project": {"died": {"$dateFromString": {"dateString": "$died"}},
                  "born": {"$dateFromString": {"dateString": "$born"}},
                  "_id": 0}},
    {"$project": {"years": {"$floor": {"$divide": [{"$subtract": ["$died", "$born"]},
                                                   31557600000]}}}}, # 1000 * 60 * 60 * 24 * 365.25 ms
    {"$bucket": {"groupBy": "$years",
                 "boundaries": list(range(30, 120, 10))}}
]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('*********************************************************')
print('** 04.14 "...it\'s the life in your years"')
print('*********************************************************')
pipeline = [
    {"$match": {"died": {"$gt": "1700"}, 
                "born": {"$gt": "1700"}}},
    {"$addFields": {"bornArray": {"$split": ["$born", "-"]},
                    "diedArray": {"$split": ["$died", "-"]}}},
    {"$addFields": {"born": {"$cond": [{"$in": ["00", "$bornArray"]},
                                       {"$concat": [{"$arrayElemAt": ["$bornArray", 0]}, "-01-01"]},
                                       "$born"]}}},
    {"$project": {"died": {"$dateFromString": {"dateString": "$died"}},
                  "born": {"$dateFromString": {"dateString": "$born"}},
                  "firstname": 1, 
                  "surname": 1,
                  "_id": 0}},
    {"$project": {"_id": 0,
                  "firstname": 1, 
                  "surname": 1,
                  "years": {"$floor": {"$divide": [{"$subtract": ["$died", "$born"]},
                                                   31557600000]}}}}, # 1000 * 60 * 60 * 24 * 365.25 ms
    # For exploration purpose
    {'$limit': 3}
]
data = list(db.laureates.aggregate(pipeline))
pprint(data)

print('*********************************************************')
print('** 04.15 How many prizes were awarded to immigrants?')
print('*********************************************************')
# Finding documents with more than one affiliations
db.laureates.count_documents({'prizes.affiliations.1': {'$exists': True}})

# How many documents decompressing until prizes.affiliations
print('Exploring...')
pipeline = [# Assure only data belong to person
            {'$match': {'gender': {'$ne': 'org'}}},
            # Select only needed data
            {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            {'$unwind': "$prizes.affiliations"},
            # Add a new field
            {"$addFields": {"prizeACExist": {'$ne': ["$prizes.affiliations.country", []]}}},
            {"$addFields": {"affilCountrySameAsBorn": {"$cond": [{'$ne': ["$prizes.affiliations.country", []]},
                                                                 {"$gte": [{"$indexOfBytes": ["$prizes.affiliations.country",
                                                                                              "$bornCountry"]}, 0]},
                                                                 False]}}},
            # Reproject the data
            {"$project": {"bornCountry": 1, 
                          "prizes.affiliations.country": 1, 
                          'prizeACExist': 1,
                          'affilCountrySameAsBorn': 1,
                          '_id': 0}},
]
data = list(db.laureates.aggregate(pipeline))
print('\nNumber of docs (decompressing until prizes.affiliations):', len(data))
print('3 first docs as example:')
pprint(data[:3])

pipeline = [# Assure only data belong to person
            {'$match': {'gender': {'$ne': 'org'}}},
            # Select only needed data
            {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            # Add a new field
            {"$addFields": {"prizeACExist": {'$ne': ["$prizes.affiliations.country", []]}}},
            {"$addFields": {"affilCountrySameAsBorn": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
            # Reproject the data
            {"$project": {"bornCountry": 1, 
                          "prizes.affiliations.country": 1, 
                          'prizeACExist': 1,
                          'affilCountrySameAsBorn': 1,
                          '_id': 0}},
]
data = list(db.laureates.aggregate(pipeline))
print('\nNumber of docs (decompressing until prizes):', len(data))
print('3 first docs as example:')
pprint(data[:3])

# For example, the prize affiliation country "Germany" should match the country of birth "Prussia (now Germany).
# First approximation (without book help)
print('\nWithout book help...')
pipeline = [# Assure only data belong to person
            {'$match': {'gender': {'$ne': 'org'}}},
            # Select only needed data
            {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            # Add a new field
            {"$addFields": {"prizeACExist": {'$ne': ["$prizes.affiliations.country", []]}}},
            {"$addFields": {"affilCountrySameAsBorn": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
            # Reproject the data
            {"$project": {"bornCountry": 1, 
                          "prizes.affiliations.country": 1, 
                          'prizeACExist': 1,
                          'affilCountrySameAsBorn': 1,
                          'bornCountryInAffiliations': 1,
                          '_id': 0}},
            # Filtering the data with no Affiliation (special cases in data)
            {'$match': {'prizeACExist': False}},
            # For exploration purpose
            {'$limit': 3}
] 
data = list(db.laureates.aggregate(pipeline))
print('Exploring...')
pprint(data)


pipeline = [# Assure only data belong to person
            {'$match': {'gender': {'$ne': 'org'}}},
            # Select only needed data
            {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1, '_id': 0}},
            # Ensure a single prize affiliation country per pipeline document
            {'$unwind': "$prizes"},
            # Reproject the data
            {"$project": {"affilCountrySameAsBorn": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
            # Count by "$affilCountrySameAsBorn" value (True or False)
            {"$group": {"_id": "$affilCountrySameAsBorn",
                        "count": {"$sum": 1}}},
            # Show only no affilliation count
            {'$match': {'_id': False}}
] 
data = list(db.laureates.aggregate(pipeline))
print('\nFinal result...')
pprint(data)

print('\nBase on lesson...')
print('With $in operator')
pipeline = [
    # Limit results to people; project needed fields; unwind prizes
    {'$match': {'gender': {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
            
    # Count prizes with no country-of-birth affiliation
    {"$addFields": {"bornCountryInAffiliations": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
    {'$match': {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]
data = list(db.laureates.aggregate(pipeline))
print(f'With $addField: {data}')

pipeline = [
    # Limit results to people; project needed fields; unwind prizes
    {'$match': {'gender': {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
            
    # Count prizes with no country-of-birth affiliation
    {"$project": {"bornCountryInAffiliations": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
    {'$match': {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]
data = list(db.laureates.aggregate(pipeline))
print(f'With $project instead: {data}')

print('\nBase on lesson...')
print('With $in operator')
pipeline = [
    # Limit results to people; project needed fields; unwind prizes
    {'$match': {'gender': {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
            
    # Count prizes with no country-of-birth affiliation
    {"$addFields": {"bornCountryInAffiliations": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
    {'$match': {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]
data = list(db.laureates.aggregate(pipeline))
print(f'With $addField: {data}')

pipeline = [
    # Limit results to people; project needed fields; unwind prizes
    {'$match': {'gender': {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
            
    # Count prizes with no country-of-birth affiliation
    {"$project": {"bornCountryInAffiliations": {"$in": ['$bornCountry', "$prizes.affiliations.country"]}}},
    {'$match': {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]
data = list(db.laureates.aggregate(pipeline))
print(f'With $project instead: {data}')

print('*********************************************************')
print('** 04.16 Refinement: filter out "unaffiliated" people')
print('*********************************************************')
pipeline = [
    {"$match": {"gender": {"$ne": "org"}}},
    {"$project": {"bornCountry": 1, "prizes.affiliations.country": 1}},
    {"$unwind": "$prizes"},
    {"$addFields": {"bornCountryInAffiliations": {"$in": ["$bornCountry", "$prizes.affiliations.country"]}}},
    {"$match": {"bornCountryInAffiliations": False}},
    {"$count": "awardedElsewhere"},
]

# Construct the additional filter stage
added_stage = {"$match": {'prizes.affiliations.country': {'$in': db.laureates.distinct('prizes.affiliations.country')}}}

# Insert this stage into the pipeline
pipeline.insert(3, added_stage)
print(list(db.laureates.aggregate(pipeline)))

print('*********************************************************')
print('** 04.17 Wrap-Up')
print('*********************************************************')
print('END')
print('*********************************************************')