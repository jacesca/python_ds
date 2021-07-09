#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from datetime import date
from datetime import timedelta

from collections import Counter
from collections import defaultdict
import pandas as pd


# In[2]:


# Read data
florida_hurricane_dates = pd.read_csv('datasets/capital-onebike.csv') #list of dates
print(type(florida_hurricane_dates))
florida_hurricane_dates[:5]


# # 1. Dates and Calendars
# 
# Hurricanes (also known as cyclones or typhoons) hit the U.S. state of Florida several times per year. To start off this course, you'll learn how to work with date objects in Python, starting with the dates of every hurricane to hit Florida since 1950. You'll learn how Python handles dates, common date operations, and the right way to format dates to avoid confusion.

# # <font color=darkred>1.1 Dates in Python</font>
# 
# 1. Dates in Python
# >Hi! My name is Max Shron, I will be your instructor for this course on working with dates and times in Python. Dates are everywhere in data science. Stock prices go up and down, experiments begin and end, people are born, politicians take votes, and on and on. All these events happen at a particular point in time. Knowing how to analyze data over time is a core data science skill.
# 
# 2. Course overview
# >This course is divided into four chapters. The first chapter will be about working with dates and calendars. In chapter two, we will add time into the mix, and combine dates and times. In chapter three, we'll tackle one of the toughest parts of working with time: time zones and Daylight Saving. And finally, in chapter four, we'll connect what we've learned about working with dates and times to explore how Pandas can make answering even complex questions about dates much easier.
# 
# 3. Dates in Python
# >Let's begin. Python has a special date class, called "date", which you will use to represent dates. A date, like a string, or a number, or a numpy array, has special rules for creating it and methods for working with it. In this lesson, we're going to discuss creating dates and extracting some basic information out of them.
# 
# 4. Why do we need a date class in Python?
# >Why do we need a special date class? Let's have a look. To understand how dates work, in this chapter you're going to be exploring 67 years of Hurricane landfalls in the U.S. state of Florida. two_hurricanes is a list with the dates of two hurricanes represented as strings: the last 2016 hurricane (on October 7th, 2016) and the first 2017 hurricane (on June 21st, 2017). The dates are represented in the U.S. style, with the month, then the day, then the year. Suppose you want to do something interesting with these dates. How would you figure out how many days had elapsed between them? How would you check that they were ordered from earliest to latest? How would you know which day of the week each was? Doing these things manually would be challenging, but Python makes all of them easy. By the end of this chapter, you'll know how to do each of these things yourself.
# 
# 5. Creating date objects
# >To create a date object, we start by importing the date class. The collection of date and time-related classes are stored in the "datetime" package. We create a date using the date() function. Here we've created dates corresponding to the two hurricanes, now as Python date objects. The inputs to date() are the year, month, and day. The first date is October 7, 2016, and the second date is June 21, 2017. The order is easy to remember: it goes from the biggest to smallest. Year, month, day. Later in this chapter, you'll create dates directly from lists of strings, but in this lesson, you're going to stick to creating dates by hand or using lists of already created dates.
# 
# 6. Attributes of a date
# >You can access individual components of a date using the date's attributes. You can access the year of the date using the year attribute, like so, and the result is 2016. Similarly, you can access the month and day using the month and day attributes like so.
# 
# 7. Finding the weekday of a date
# >You can also ask Python to do more complicated work. Here we call the weekday() method on the date, and see that the weekday is 4. What does 4 mean here? Python counts weekdays from 0, starting on Monday. 1 is Tuesday, 2 is Wednesday, and so on, up to 6 being a Sunday. This date was a Friday.
# 
# 8. Dates in Python
# >In the next few exercises, you'll implement what you've seen in this video to see how much you can already do!

# In[3]:


# Creating date objects
two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
two_hurricanes_dates


# In[4]:


# Attributes of a date
print(two_hurricanes_dates[0].year)
print(two_hurricanes_dates[0].month)
print(two_hurricanes_dates[0].day)


# In[5]:


# Finding the weekday of a date
print(two_hurricanes_dates[0].weekday()) #0 is monday


# # <font color=darkred>1.2 Which day of the week?</font>
# 
# Hurricane Andrew, which hit Florida on August 24, 1992, was one of the costliest and deadliest hurricanes in US history. Which day of the week did it make landfall?
# 
# Let's walk through all of the steps to figure this out.
# 
# **Instructions**
# 1. Import date from datetime.
# 2. Create a date object for August 24, 1992.
# 3. Now ask Python what day of the week Hurricane Andrew hit (remember that Python counts days of the week starting from Monday as 0, Tuesday as 1, and so on).
# 
# **Results**
# 
# <font color=darkgreen>Great! What day does the week begin for you? It depends where you are from! In the United States, Canada, and Japan, Sunday is often considered the first day of the week. Everywhere else, it usually begins on Monday.</font>

# In[6]:


# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())


# # <font color=darkred>1.3 How many hurricanes come early?</font>
# 
# In this chapter, you will work with a list of the hurricanes that made landfall in Florida from 1950 to 2017. There were 235 in total. Check out the variable florida_hurricane_dates, which has all of these dates.
# 
# Atlantic hurricane season officially begins on June 1. How many hurricanes since 1950 have made landfall in Florida before the official start of hurricane season?
# 
# **Instructions**
# 
# - Complete the for loop to iterate through florida_hurricane_dates.
# - Complete the if statement to increment the counter (early_hurricanes) if the hurricane made landfall before June.
# 
# **Results**
# 
# <font color=darkgreen>Great! Using this same pattern, you could do more complex counting by using .day, .year, .weekday(), and so on.</font>

# In[7]:


len(florida_hurricane_dates)


# In[8]:


# Counter for how many before June 1
early_hurricanes = 0

# We loop over the dates
for hurricane in florida_hurricane_dates:
  # Check if the month is before June (month number 6)
  if hurricane.month < 6:
    early_hurricanes = early_hurricanes + 1
    
print(early_hurricanes)


# In[ ]:


display(sum([1 if h.month < 6 else 0 for h in florida_hurricane_dates]))
display(sum([h.month<6 for h in florida_hurricane_dates]))


# # <font color=darkred>1.4 Math with dates</font>
# 
# 1. Math with Dates
# >In the last lesson, we discussed how to create date objects and access their attributes. In this lesson, we're going to talk about how to do math with dates: counting days between events, moving forward or backward by a number of days, putting them in order, and so on.
# 
# 2. Math with dates
# >Let's take a step back. Think back to when you first learned arithmetic. You probably started with something like this: a number line. This one has the numbers 10 through 16 on it. A number line tells you what order the numbers go in, and how far apart numbers are from each other. Let's pick two numbers, 11 and 14, and represent them in Python as the variables a and b, respectively. We'll put them into a list, l. Python can tell us which number in this list is the least, using the min() function. min stands for the minimum. In this case, 11 is the lowest number in the list, so we get 11.
# 
# 3. Math with dates
# >We can also subtract numbers. When you subtract two numbers, in this case subtracting 11 from 14, the result is 3. Said another way, if we took three steps from 11, we would get 14.
# 
# 4. Math with dates
# >Now let's think about how this applies to dates. Let's call this line a calendar line, instead of a number line. Each dot on this calendar line corresponds to a particular day.
# 
# 5. Math with dates
# >Let's put two dates onto this calendar line: November 5th, 2017, and December 4th, 2017. Let's represent this in Python. We start by importing the date class from the datetime package. We create two date objects: d1 is November 5th, 2017, and d2 is December 4th, 2017. As before, we put them into a list, l. What Python is doing under the hood, so to speak, is not that different from putting the dates onto a calendar line. For example, if we call min of l, we again get the "least" date, which means the earliest one. In this case, that's November 5th, 2017.
# 
# 6. Math with dates
# >And just like numbers, we can subtract two dates. When we do this, we get an object of type "timedelta". Timedeltas give us the elapsed time between events. If you access the days attribute of this object, you can obtain the number of days between the two dates.
# 
# 7. Math with dates
# >We can also use a timedelta in the other direction. First, let's import timedelta from datetime. Next, we create a 29-day timedelta, by passing days=29 to timedelta(). Now when we add td to our original date we get back December 4th, 2017. Python handled the fact that November has 30 days in it for us, without us having to remember which months are 30 day months, 31 day months, or 28 day months.
# 
# 8. Incrementing variables with +=
# >Finally a quick side note: we will use the "plus-equals" operation a number of times in the rest of the course, so we should discuss it. If you aren't familiar with it, you can see how it works here. On the left-hand side, we create a variable x, set it to zero. If we set x equal to x + 1, we increment x by 1. Similarly, on the right-hand side, we set x = 0, and then we increment it with x += 1. It has the same effect, and we'll use it all the time for counting.
# 
# 9. Let's Practice!
# >We talked about how date objects are very similar to numbers, and how you can subtract them to get a timedelta, or add a timedelta to a date to get another date. We also briefly touched on the += operator. It's time for you to practice these concepts.

# In[ ]:


# Create our dates
d1 = date(2017, 11, 5)
d2 = date(2017, 12, 4)
l = [d1, d2]
l


# In[ ]:


# Math with dates
print(min(l))


# In[ ]:


# Subtract two dates
delta = d2 - d1
print(delta.days)
delta


# In[ ]:


# Create a 29 day timedelta
td = timedelta(days=29)
print(d1 + td)


# # <font color=darkred>1.5 Subtracting dates</font>
# 
# Python date objects let us treat calendar dates as something similar to numbers: we can compare them, sort them, add, and even subtract them. This lets us do math with dates in a way that would be a pain to do by hand.
# 
# The 2007 Florida hurricane season was one of the busiest on record, with 8 hurricanes in one year. The first one hit on May 9th, 2007, and the last one hit on December 13th, 2007. How many days elapsed between the first and last hurricane in 2007?
# 
# **Instructions**
# - Import date from datetime.
# - Create a date object for May 9th, 2007, and assign it to the start variable.
# - Create a date object for December 13th, 2007, and assign it to the end variable.
# - Subtract start from end, to print the number of days in the resulting timedelta object.
# 
# **Results**
# 
# <font color=darkgreen>Good job! One thing to note: be careful using this technique for historical dates hundreds of years in the past. Our calendar systems have changed over time, and not every date from then would be the same day and month today.</font>

# In[ ]:


# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)


# # <font color=darkred>1.6 Counting events per calendar month</font>
# 
# Hurricanes can make landfall in Florida throughout the year. As we've already discussed, some months are more hurricane-prone than others.
# 
# Using florida_hurricane_dates, let's see how hurricanes in Florida were distributed across months throughout the year.
# 
# We've created a dictionary called hurricanes_each_month to hold your counts and set the initial counts to zero. You will loop over the list of hurricanes, incrementing the correct month in hurricanes_each_month as you go, and then print the result.
# 
# **Instructions**
# 
# - Within the for loop:
# - Assign month to be the month of that hurricane.
# - Increment hurricanes_each_month for the relevant month by 1.
# 
# **Results**
# 
# <font color=darkgreen>Success! This illustrated a generally useful pattern for working with complex data: creating a dictionary, performing some operation on each element, and storing the results back in the dictionary.</font>

# In[ ]:


# A dictionary to count hurricanes per calendar month
hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0, 7: 0, 8:0, 9:0, 10:0, 11:0, 12:0}

# Loop over all hurricanes
for hurricane in florida_hurricane_dates:
  # Pull out the month
  month = hurricane.month
  # Increment the count in your dictionary by one
  hurricanes_each_month[month] += 1

hurricanes_each_month


# In[ ]:


# Simplest form
dict(sorted(Counter(pd.DatetimeIndex(florida_hurricane_dates).month).items()))


# In[ ]:


# Adding a default value
huracane_per_month = defaultdict(int, dict(sorted(Counter(pd.DatetimeIndex(florida_hurricane_dates).month).items())))

display(huracane_per_month)
display(huracane_per_month[0])
display(huracane_per_month)


# # <font color=darkred>1.7 Putting a list of dates in order</font>
# 
# Much like numbers and strings, date objects in Python can be put in order. Earlier dates come before later ones, and so we can sort a list of date objects from earliest to latest.
# 
# What if our Florida hurricane dates had been scrambled? We've gone ahead and shuffled them so they're in random order and saved the results as dates_scrambled. Your job is to put them back in chronological order, and then print the first and last dates from this sorted list.
# 
# **Instructions**
# 
# 1. Print the first and last dates in dates_scrambled.
# 2. Sort dates_scrambled using Python's built-in sorted() method, and save the results to dates_ordered.
# 3. Print the first and last dates in dates_ordered.
# 
# **Results**
# 
# <font color=darkgreen>Excellent! You can use sorted() on several data types in Python, including sorting lists of numbers, lists of strings, or even lists of lists, which by default are compared on the first element.</font>

# In[ ]:


# Print the first and last scrambled dates
print(florida_hurricane_dates[0])
print(florida_hurricane_dates[-1])


# In[ ]:


# Put the dates in order
dates_ordered = sorted(florida_hurricane_dates)

# Print the first and last ordered dates
print(dates_ordered[0])
print(dates_ordered[-1])


# # <font color=darkred>1.8 Turning dates into strings</font>
# 
# **1. Turning dates into strings**
# >Python has a very flexible set of tools for turning dates back into strings to be easily read. We want to put dates back into strings when, for example, we want to print results, but also if we want to put dates into filenames, or if we want to write dates out to CSV or Excel files.
# 
# **2. ISO 8601 format**
# >For example, let's create a date and see how Python prints it by default. As before, we import date from datetime and let's again create an object for November 5th, 2017. When we ask Python to print the date, it prints the year, day and then the month, separated by dashes, and always with two digits for the day and month. In the comment, you can see I've noted this as YYYY-MM-DD; four digit year, two digit month, and two digit day of the month. This default format is also known as ISO format, or ISO 8601 format, after the international standard ISO 8601 that it is based on. ISO 8601 strings are always the same length since month and day are written with 0s when they are less than 10. We'll talk about another advantage of ISO 8601 in a moment. If we want the ISO representation of a date as a string, say to write it to a CSV file instead of just printing it, you can call the isoformat() method. In this example, we put it inside a list so you can see that it creates a string.
# 
# **3. ISO 8601 format**
# >The ISO 8601 format has another nice advantage. To demonstrate, we've created a variable called some_dates and represented two dates here as strings: January 1, 2000, and December 31, 1999. Dates formatted as ISO 8601 strings sort correctly. When we print the sorted version of this list, the earlier day is first, and the later date is second. For example, if we use ISO 8601 dates in filenames, they can be correctly sorted from earliest to latest. If we had month or day first, the strings would not sort in chronological order.
# 
# **4. Every other format**
# >If you don't want to put dates in ISO 8601 format, Python has a flexible set of options for representing dates in other ways, using the strftime() method.
# 
# **5. Every other format: strftime**
# >strftime() works by letting you pass a "format string" which Python uses to format your date. Let's see an example. We again create an example date of January 5th, 2017. We then call strftime() on d, with the format string of % capital Y. Strftime reads the % capital Y and fills in the year in this string for us. strftime() though is very flexible: we can give it arbitrary strings with % capital Y in them for the format string, and it will stick the year in. For example, we can use the format string of "Year is %Y".
# 
# **6. Every other format: strftime**
# >Strftime has other placeholders besides %Y: % lowercase m gives the month, and % lowercase d gives the day of the month. Using these, we can represent dates in arbitrary formats for whatever our needs are.
# 
# **7. Turning dates into strings**
# >In this lesson, we discussed how Python can represent a date as a string. We emphasized the importance and utility of ISO 8601 format, but also introduced strftime(), which lets you turn dates in a wide variety of strings depending on your needs.

# In[ ]:


# Example date
d = date(2017, 11, 5)

# ISO format: YYYY-MM-DD
print(d)


# In[ ]:


# Express the date in ISO 8601 format and put it in a list
print( [d.isoformat()] )


# In[ ]:


# Every other format: strftime
print(d.strftime("%Y"))

# Format string with more text in it
print(d.strftime("Year is %Y"))

# Format: YYYY/MM/DD
print(d.strftime("%Y/%m/%d"))


# # <font color=darkred>1.9 Printing dates in a friendly format</font>
# 
# Because people may want to see dates in many different formats, Python comes with very flexible functions for turning date objects into strings.
# 
# Let's see what event was recorded first in the Florida hurricane data set. In this exercise, you will format the earliest date in the florida_hurriance_dates list in two ways so you can decide which one you want to use: either the ISO standard or the typical US style.
# 
# **Instructions**
# 
# - Assign the earliest date in florida_hurricane_dates to first_date.
# - Print first_date in the ISO standard. For example, December 1st, 2000 would be "2000-12-01".
# - Print first_date in the US style, using .strftime(). For example, December 1st, 2000 would be "12/1/2000" .
# 
# **Results**
# 
# <font color=darkgreen>Correct! When in doubt, use the ISO format for dates. ISO dates are unambiguous. And if you sort them 'alphabetically', for example, in filenames, they will be in the correct order.</font>

# In[ ]:


# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)


# # <font color=darkred>1.10 Representing dates in different ways</font>
# 
# date objects in Python have a great number of ways they can be printed out as strings. In some cases, you want to know the date in a clear, language-agnostic format. In other cases, you want something which can fit into a paragraph and flow naturally.
# 
# Let's try printing out the same date, August 26, 1992 (the day that Hurricane Andrew made landfall in Florida), in a number of different ways, to practice using the .strftime() method.
# 
# A date object called andrew has already been created.
# 
# **Instructions**
# 
# - Print andrew in the format 'YYYY-MM'.
# - Print andrew in the format 'MONTH (YYYY)', using %B for the month's full name, which in this case will be August.
# - Print andrew in the format 'YYYY-DDD' (where DDD is the day of the year) using %j.
# 
# **Results**
# 
# <font color=darkgreen>Nice! Pick the format that best matches your needs. For example, astronomers usually use the 'day number' out of 366 instead of the month and date, to avoid ambiguities between languages.</font>

# In[ ]:


# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-MM'
print(andrew.strftime("%Y-%m"))

# Print the date in the format 'MONTH (YYYY)'
print(andrew.strftime("%B (%Y)"))

# Print the date in the format 'YYYY-DDD'
print(andrew.strftime("%Y-%j"))


# # Aditional material
# 
# - Datacamp course: https://learn.datacamp.com/courses/extreme-gradient-boosting-with-xgboost
# - Xgboost documentation: https://xgboost.readthedocs.io/en/latest/
# - sklearn.tree.DecisionTreeClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
