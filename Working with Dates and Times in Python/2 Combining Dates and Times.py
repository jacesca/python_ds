#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
from datetime import datetime
from collections import Counter

import pandas as pd
import expectexception
from datetime import timedelta


# In[2]:


# Read data
bikes_data = pd.read_csv('datasets/capital-onebike.csv') #list of dates
print(bikes_data.shape)
bikes_data.columns = bikes_data.columns.str.lower().str.replace(' ','_')
bikes_data.head()


# ### <font color=blue>Transform data to datetime</font>

# In[3]:


# Transforming data to Timestamp using pd.to_datetime (First choice)
bikes_data['end'] = pd.to_datetime(bikes_data['end_date'])
bikes_data['start'] = pd.to_datetime(bikes_data['start_date'])
bikes_data[['end','start']].to_dict(orient='records')[:2]


# In[4]:


# Transforming data to datetime
display(bikes_data['end'].dt.to_pydatetime()[:2])

display(pd.to_datetime(bikes_data['end_date']).dt.to_pydatetime()[:2])

# Transforming data to date
display([d.date() for d in pd.to_datetime(bikes_data['end'])][:3])


# In[5]:


# Transforming to dict of datetime (Second choice)
[{'end'  : datetime.strptime(row['end_date'], "%Y-%m-%d %H:%M:%S"),
  'start': datetime.fromisoformat(row['start_date'])} for i, row in bikes_data.iterrows()][:2]


# ### <font color=blue>Global variables</font>

# In[6]:


# Transforming to dict of datetime (final choice)
onebike_datetimes = [{'end'  : pd.to_datetime(row['end_date']).to_pydatetime(),
                      'start': pd.to_datetime(row['start_date']).to_pydatetime()} for i, row in bikes_data.iterrows()]
onebike_datetimes[:2]


# In[7]:


onebike_datetime_strings = list(bikes_data[['end_date', 'start_date']].to_records(index=False))
onebike_datetime_strings[:2]


# # 2. Combining Dates and Times
# 
# Bike sharing programs have swept through cities around the world -- and luckily for us, every trip gets recorded! Working with all of the comings and goings of one bike in Washington, D.C., you'll practice working with dates and times together. You'll parse dates and times from text, analyze peak trip times, calculate ride durations, and more.

# # <font color=darkred>2.1 Dates and times</font>
# 
# **1. Adding time to the mix**
# >In this chapter, you are going to move from only working with dates to working with both dates and times: the calendar day AND the time on the clock within that day.
# 
# **2. Dates and Times**
# >As always, let's start with an example. Here is an example of a date and a time together: October 1, 2017, at 3:23:25 PM. Unlike before, where we were only working with the date, we're now going to also include the time. Let's see how to represent this in Python.
# 
# **3. Dates and Times**
# >The first thing we have to do is import the datetime class from the datetime package. Ideally, these would have different names, but unfortunately for historical reasons they have the same name. This is just something to get used to.
# 
# **4. Dates and Times**
# >We're going to create a datetime called "dt" and populate the fields together.
# 
# **5. Dates and Times**
# >The first three arguments to the datetime class are exactly the same as the date class. Year, then month, then day, each as a number.
# 
# **6. Dates and Times**
# >Next, we fill in the hour. Computers generally use 24 hour time, meaning that 3 PM is represented as hour 15 of 24.
# 
# **7. Dates and Times**
# >We put in the minutes, 23 out of 60.
# 
# **8. Dates and Times**
# >And finally, the seconds. October 1, 2017 at 3:23:25PM is represented as a datetime in Python as 2017, 10, 1, 15, 23, 25). All of these arguments need to be whole numbers; if you want to represent .5 seconds,
# 
# **9. Dates and Times**
# >you can add microseconds to your datetime. Here we've added 500,000 microseconds, or .5 seconds. That is, Python breaks seconds down into millionths of a second for you when you need that kind of precision. If you need billionths of a second precision (which happens sometimes in science and finance) we'll cover nanoseconds when we get to Pandas at the end of this course. Python defaults to 0 microseconds if you don't include it.
# 
# **10. Dates and Times**
# >That's a lot of arguments; if it helps, you can always be more explicit and use named arguments.
# 
# **11. Replacing parts of a datetime**
# >We can also make new datetimes from existing ones by using the replace() method. For example, we can take the datetime we just made, and make a new one which has the same date but is rounded down to the start of the hour. We call dt.replace() and set minutes, seconds, and microseconds to 0. This creates a new datetime with the same values in all the other fields, but these ones changed.
# 
# **12. Capital Bikeshare**
# >Before we wrap up, let's talk about the data we will use for the rest of this course. You will be working with data from Capital Bikeshare, the oldest municipal shared bike program in the United States. Throughout the Washington, D.C. area, you will find these special bike docks, where riders can pay to take a bike, ride it, and return to this or any other station in the network. We will be following one bike, ID number "W20529", on all the trips it took in October, November, and December of 2017. Each trip consisted of a date and time when a bike was undocked from a station, then some time passed, and the date and time when W20529 was docked again.
# 
# **13. Adding time to the mix**
# >In this video, we walked through how to create datetime objects in Python. You're going to practice that in the exercises, and also work with the Capital Bikeshare data and see how we can use Python to understand the trips that W20529 took throughout the three months we're interested in.

# In[8]:


# Dates and Times
dt = datetime(2017, 10, 1, 15, 23, 25)
print(type(dt))
print(dt)


# In[9]:


dt = datetime(2017, 10, 1, 15, 23, 25, 500000)
print(dt)

dt = datetime(2017, 10, 1, 15, 23, 25, 5)
print(dt)


# In[10]:


dt_hr = dt.replace(minute=0, second=0, microsecond=0)
print(dt_hr)


# # <font color=darkred>2.2 Creating datetimes by hand</font> 
# 
# Often you create datetime objects based on outside data. Sometimes though, you want to create a datetime object from scratch.
# 
# You're going to create a few different datetime objects from scratch to get the hang of that process. These come from the bikeshare data set that you'll use throughout the rest of the chapter.
# 
# **Instructions**
# 
# - Import the datetime class.
# - Create a datetime for October 1, 2017 at 15:26:26.
# - Print the results in ISO format.
# - Create a datetime for December 31, 2017 at 15:19:13.
# - Print the results in ISO format.
# - Create a new datetime by replacing the year in dt with 1917 (instead of 2017)
# 
# **Results**
# 
# <font color=darkgreen>Well done! You can now create datetime objects.</font>

# In[11]:


# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Print the results in ISO 8601 format
print(dt.isoformat())


# In[12]:


# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Print the results in ISO 8601 format
print(dt.isoformat())


# In[13]:


# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt)
print(dt_old)


# # <font color=darkred>2.3 Counting events before and after noon</font> 
# 
# In this chapter, you will be working with a list of all bike trips for one Capital Bikeshare bike, W20529, from October 1, 2017 to December 31, 2017. This list has been loaded as onebike_datetimes.
# 
# Each element of the list is a dictionary with two entries: start is a datetime object corresponding to the start of a trip (when a bike is removed from the dock) and end is a datetime object corresponding to the end of a trip (when a bike is put back into a dock).
# 
# You can use this data set to understand better how this bike was used. Did more trips start before noon or after noon?
# 
# **Instructions**
# - Within the for loop, complete the if statement to check if the trip started before noon.
# - Within the for loop, increment trip_counts['AM'] if the trip started before noon, and trip_counts['PM'] if it started after noon.
# 
# **Results**
# 
# <font color=darkgreen>Great! It looks like this bike is used about twice as much after noon than it is before noon. One obvious follow up would be to see which hours the bike is most likely to be taken out for a ride.</font>

# In[14]:


# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}
  
# Loop over all trips
for trip in onebike_datetimes:
  # Check to see if the trip starts before noon
  if trip['start'].hour < 12:
    # Increment the counter for before noon
    trip_counts['AM'] += 1
  else:
    # Increment the counter for after noon
    trip_counts['PM'] += 1
  
print(trip_counts)


# In[15]:


# Simplify the code
Counter(['AM' if trip['start'].hour < 12 else 'PM' for trip in onebike_datetimes])


# # <font color=darkred>2.4 Printing and parsing datetimes</font>
# 
# **1. Printing and parsing datetimes**
# >Much like dates, datetimes can be printed in many ways. Python also has another trick: you can take a string and turn it directly into a datetime. Let's start with printing dates and then move on to asking Python to parse them.
# 
# **2. Printing datetimes**
# >First, let's create a datetime again. dt corresponds to December 30, 2017 at 15:19:13, the end of the last trip that W20529 takes in our data set. Just like with date objects, we use strftime() to create a string with a particular format. First, we'll just print the year, month and date, using the same format codes we used for dates. % capital Y stands for the four digit year, % lowercase m for the month, and % lowercase d for the day of the month. Now we can add in the hours, minutes and seconds. Again, we print the year, month and day, and now we add three more format codes: % capital H gives us the hour, % capital M gives us the minute, and % capital S gives us the seconds. There are also format codes for 12-hour clocks, and for printing the correct AM or PM.
# 
# **3. Printing datetimes**
# >As before, we can make these formatting strings as complicated as we need. Here's another version of the previous string.
# 
# **4. ISO 8601 Format**
# >Finally, we can use the isoformat() method, just like with dates, to get a standards-compliant way of writing down a datetime. The officially correct way of writing a datetime is the year, month, day, then a capital T, then the time in 24 hour time, followed by the minute and second. When in doubt, this is a good format to use.
# 
# **5. Parsing datetimes with strptime**
# >We can also parse dates from strings, using the same format codes we used for printing. You'll use this often when getting date and time data from the Internet since dates and times are often represented as strings. We start, as before, by importing the datetime class.
# 
# **6. Parsing datetimes with strptime**
# >The method we're going to use is called strptime(), which is short for string parse time. strptime() takes two arguments: the first argument is the string to turn into a datetime, and the second argument is the format string that says how to do it.
# 
# **7. Parsing datetimes with strptime**
# >First we pass the string we want to parse. In this case, a string representing December 30, 2017, at 15:19:13.
# 
# **8. Parsing datetimes with strptime**
# >Then we pass the format string, which as mentioned before uses the same format codes we used with strftime(). In this case, first the month, then the day, then the year, all separated by slashes, then a space, and then the hour, minutes, and seconds separated by colons. You usually need to figure this out once per data set.
# 
# **9. Parsing datetimes with strptime**
# >If we look and see what kind of object we've made, by printing the type of dt, we see that we've got a datetime. And if we print that datetime, we get a string representation of the datetime. We can see that the parsing worked correctly.
# 
# **10. Parsing datetimes with strptime**
# >We need an exact match to do a string conversion. For example, if we leave out how to parse the time, Python will throw an error. And similarly, if there is an errant comma or other symbols, strptime() will not be happy.
# 
# **11. Parsing datetimes with Pandas**
# >Finally, there is another kind of datetime you will sometimes encounter: the Unix timestamp. Many computers store datetime information behind the scenes as the number of seconds since January 1, 1970. This date is largely considered the birth of modern-style computers. To read a Unix timestamp, use the datetime.fromtimestamp() method. Python will read your timestamp and return a datetime.
# 
# **12. Printing and parsing datetimes**
# >We've just covered how to use format codes to turn datetimes into strings and strings into datetimes, and what ISO 8601 format looks like with time involved. Now you'll practice moving back and forth between strings and datetimes.

# In[16]:


# Create datetime
dt = datetime(2017, 12, 30, 15, 19, 13)

# Printing datetimes
display(dt.strftime("%Y-%m-%d"))
display(dt.strftime("%Y-%m-%d %H:%M:%S"))
display(dt.strftime("%Y-%m-%d %I:%M:%S %p"))
display(dt.strftime("%H:%M:%S on %d/%m/%Y"))

# ISO 8601 format
display(dt.isoformat())


# In[17]:


# Parsing datetimes with strptime
dt = datetime.strptime("12/30/2017 15:19:13", "%m/%d/%Y %H:%M:%S")

# What did we make?
print(type(dt))

# Print out datetime object
print(dt)


# In[18]:


get_ipython().run_cell_magic('expect_exception', 'ValueError', '# Incorrect format string\ndt = datetime.strptime("2017-12-30 15:19:13", "%Y-%m-%d")')


# In[19]:


# Parsing datetimes with Pandas - A UNIX timestamp
ts = 1514665153.0

# Convert to datetime and print
print(datetime.fromtimestamp(ts))


# # <font color=darkred>2.5 Turning strings into datetimes</font> 
# 
# When you download data from the Internet, dates and times usually come to you as strings. Often the first step is to turn those strings into datetime objects.
# 
# In this exercise, you will practice this transformation.
# 
# |**Reference**| |
# |-|-|
# |%Y|4digit year (0000-9999)|
# |%m|2digit month (1-12)|
# |%d|2 digit day (1-31)|
# |%H|2 digit hour (0-23)|
# |%M|2 digit minute (0-59)|
# |%S|2 digit second (0-59)|
# 
# **Instructions**
# - Determine the format needed to convert s to datetime and assign it to fmt.
# - Convert the string s to datetime using fmt.
# 
# 
# - Determine the format needed to convert s to datetime and assign it to fmt.
# - Convert the string s to datetime using fmt.
# 
# 
# - Determine the format needed to convert s to datetime and assign it to fmt.
# - Convert the string s to datetime using fmt.
# 
# **Results**
# 
# <font color=darkgreen>Great! Now you can parse dates in most common formats. Unfortunately, Python does not have the ability to parse non-zero-padded dates and times out of the box (such as 1/2/2018). If needed, you can use other string methods to create zero-padded strings suitable for strptime().</font>

# In[20]:


# Starting string, in YYYY-MM-DD HH:MM:SS format
s = '2017-02-03 00:00:01'

# Write a format string to parse s
fmt = '%Y-%m-%d %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# In[21]:


# Starting string, in YYYY-MM-DD format
s = '2030-10-15'

# Write a format string to parse s
fmt = '%Y-%m-%d'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# In[22]:


# Starting string, in MM/DD/YYYY HH:MM:SS format
s = '12/15/1986 08:00:00'

# Write a format string to parse s
fmt = '%m/%d/%Y %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# # <font color=darkred>2.6 Parsing pairs of strings as datetimes</font> 
# 
# Up until now, you've been working with a pre-processed list of datetimes for W20529's trips. For this exercise, you're going to go one step back in the data cleaning pipeline and work with the strings that the data started as.
# 
# Explore onebike_datetime_strings in the IPython shell to determine the correct format. datetime has already been loaded for you.
# 
# |**Reference**| |
# |-|-|
# |%Y|4digit year (0000-9999)|
# |%m|2digit month (1-12)|
# |%d|2 digit day (1-31)|
# |%H|2 digit hour (0-23)|
# |%M|2 digit minute (0-59)|
# |%S|2 digit second (0-59)|
# 
# **Instructions**
# - Outside the for loop, fill out the fmt string with the correct parsing format for the data.
# - Within the for loop, parse the start and end strings into the trip dictionary with start and end keys and datetime objects for values.
# 
# **Results**
# 
# <font color=darkgreen>Excellent! Now you know how to process lists of strings into a more useful structure. If you haven't come across this approach before, many complex data cleaning tasks follow this same format: start with a list, process each element, and add the processed data to a new list.</font>

# In[23]:


# Write down the format string
fmt = "%Y-%m-%d %H:%M:%S"

# Initialize a list for holding the pairs of datetime objects
bike_datetimes = []

# Loop over all trips
for (start, end) in onebike_datetime_strings:
  trip = {'start': datetime.strptime(start, fmt),
          'end': datetime.strptime(end, fmt)}
  
  # Append the trip
  bike_datetimes.append(trip)
bike_datetimes[:2]


# In[24]:


#Simplify the code
[{'start': datetime.strptime(start, fmt), 'end': datetime.strptime(end, fmt)} for start, end in onebike_datetime_strings][:2]


# # <font color=darkred>2.7 Recreating ISO format with strftime()</font> 
# 
# In the last chapter, you used strftime() to create strings from date objects. Now that you know about datetime objects, let's practice doing something similar.
# 
# Re-create the .isoformat() method, using .strftime(), and print the first trip start in our data set.
# 
# |**Reference**| |
# |-|-|
# |%Y|4digit year (0000-9999)|
# |%m|2digit month (1-12)|
# |%d|2 digit day (1-31)|
# |%H|2 digit hour (0-23)|
# |%M|2 digit minute (0-59)|
# |%S|2 digit second (0-59)|
# 
# **Instructions**
# - Complete fmt to match the format of ISO 8601.
# - Print first_start with both .isoformat() and .strftime(); they should match.
# 
# **Results**
# 
# <font color=darkgreen>Awesome! There are a wide variety of time formats you can create with strftime(), depending on your needs. However, if you don't know exactly what you need, .isoformat() is a perfectly fine place to start.</font>

# In[25]:


# Pull out the start of the first trip
first_start = onebike_datetimes[0]['start']
print('Just the variable:', first_start)

# Format to feed to strftime()
fmt = "%Y-%m-%dT%H:%M:%S"

# Print out date with .isoformat(), then with .strftime() to compare
print('With .isoformat():', first_start.isoformat())
print('With .strftime() :', first_start.strftime(fmt))


# # <font color=darkred>2.8 Unix timestamps</font> 
# 
# Datetimes are sometimes stored as Unix timestamps: the number of seconds since January 1, 1970. This is especially common with computer infrastructure, like the log files that websites keep when they get visitors.
# 
# **Instructions**
# 
# - Complete the for loop to loop over timestamps.
# - Complete the code to turn each timestamp ts into a datetime.
# 
# **Results**
# 
# <font color=darkgreen>Nice! The largest number that some older computers can hold in one variable is 2147483648, which as a Unix timestamp is in January 2038. On that day, many computers which haven't been upgraded will fail. Hopefully, none of them are running anything critical!</font>

# In[26]:


# Starting timestamps
timestamps = [1514665153, 1514664543]

# Datetime objects
dts = []

# Loop
for ts in timestamps:
  dts.append(datetime.fromtimestamp(ts))
  
# Print results
print(dts)


# # <font color=darkred>2.9 Working with durations</font>
# 
# **1. Working with durations**
# >Much like dates, datetimes have a kind of arithmetic; we can compare them, subtract them, and add intervals to them. Because we are working with both days and times, the logic for durations is a little more complicated, but not by much. Let's have a look.
# 
# **2. Working with durations**
# >Just as with dates, to get a sense for what's going on we put our datetimes on a timeline. These two datetimes here correspond to the start and end of one ride in our data set.
# 
# **3. Working with durations**
# >To follow along in Python, we'll load these two in as "start" and "end". When we subtract datetimes, we get a timedelta. A timedelta represents what is called a duration: the elapsed time between events.
# 
# **4. Working with durations**
# >When we call the method total_seconds(), we get back the number of seconds that our timedelta represents. In this case, 1450 seconds elapsed between our start and end. 1450 seconds is 24 minutes and 10 seconds.
# 
# **5. Creating timedeltas**
# >You can also create a timedelta by hand. You start by importing timedelta from datetime. To create a timedelta, you specify the amount of time which has elapsed. For example, we make delta1, a timedelta which corresponds to a one second duration.
# 
# **6. Creating timedeltas**
# >Now when we add delta1 to start, we see that we get back a datetime which is one second later.
# 
# **7. Creating timedeltas**
# >We also create a timedelta, delta2, which is one day and one second in duration. Now when we add it to start, we get a new datetime which is the next day and one second later. Timedeltas can be created with any number of weeks, days, minutes, hours, seconds, or microseconds, and can be as small as a microsecond or as large as 2.7 million years.
# 
# **8. Negative timedeltas**
# >Timedeltas can also be negative. For example, if we create delta3, whose argument is -1 weeks, and we add it to start we get a datetime corresponding to one week earlier.
# 
# **9. Negative timedeltas**
# >We can also subtract a positive timedelta and get the same result. We create delta4, which corresponds to a 1 week duration, and we subtract it from start. As you can see, we get the same answer as when we added a negative timedelta.
# 
# **10. Working with durations**
# >In this lesson, we showed how to create timedeltas, which represent a duration of time. You can create timedeltas by subtracting two datetimes, which tells you how much time has elapsed between events. You can also create a timedelta directly, then add and subtract timedeltas from datetimes to make new datetimes. In the next few exercises, you'll see how these pieces can be combined together to answer more complex questions.

# In[27]:


# Create example datetimes
start = datetime(2017, 10, 8, 23, 46, 47)
end = datetime(2017, 10, 9, 0, 10, 57)
print('Start     :', start)
print('End       :', end)

# Subtract datetimes to create a timedelta
duration = end - start
print('Between   :', duration)

# Subtract datetimes to create a timedelta
print('In seconds:', duration.total_seconds())


# In[28]:


# Create a timedelta
delta1 = timedelta(seconds=1)
print(delta1)

print(start)

# One second later
print(start + delta1)


# In[29]:


# Create a one day and one second timedelta
delta2 = timedelta(days=1, seconds=1)
print(delta2)

print(start)

# One day and one second later
print(start + delta2)


# In[30]:


# Create a negative timedelta of one week
delta3 = timedelta(weeks=-1)
print(delta3)

print(start)

# One week earlier
print(start + delta3)


# In[31]:


# Same, but we'll subtract this time
delta4 = timedelta(weeks=1)
print(delta4)

print(start)

# One week earlier
print(start - delta4)


# # <font color=darkred>2.10 Turning pairs of datetimes into durations</font> 
# 
# When working with timestamps, we often want to know how much time has elapsed between events. Thankfully, we can use datetime arithmetic to ask Python to do the heavy lifting for us so we don't need to worry about day, month, or year boundaries. Let's calculate the number of seconds that the bike was out of the dock for each trip.
# 
# Continuing our work from a previous coding exercise, the bike trip data has been loaded as the list onebike_datetimes. Each element of the list consists of two datetime objects, corresponding to the start and end of a trip, respectively.
# 
# **Instructions**
# - Within the loop:
#     - Use arithmetic on the start and end elements to find the length of the trip
#     - Save the results to trip_duration.
#     - Calculate trip_length_seconds from trip_duration.
# 
# **Results**
# 
# <font color=darkgreen>Success! Remember that timedelta objects are represented in Python as a number of days and seconds of elapsed time. Be careful not to use .seconds on a timedelta object, since you'll just get the number of seconds without the days!</font>

# In[32]:


# Initialize a list for all the trip durations
onebike_durations = []

for trip in onebike_datetimes:
  # Create a timedelta object corresponding to the length of the trip
  trip_duration = trip['end'] - trip['start']
  
  # Get the total elapsed seconds in trip_duration
  trip_length_seconds = trip_duration.total_seconds()
  
  # Append the results to our list
  onebike_durations.append(trip_length_seconds)

onebike_durations[:2]


# In[33]:


# Simplify the code
[(trip['end'] - trip['start']).total_seconds() for trip in onebike_datetimes][:3]


# # <font color=darkred>2.11 Average trip time</font> 
# 
# W20529 took 291 trips in our data set. How long were the trips on average? We can use the built-in Python functions sum() and len() to make this calculation.
# 
# Based on your last coding exercise, the data has been loaded as onebike_durations. Each entry is a number of seconds that the bike was out of the dock.
# 
# **Instructions**
# - Calculate total_elapsed_time across all trips in onebike_durations.
# - Calculate number_of_trips for onebike_durations.
# - Divide total_elapsed_time by number_of_trips to get the average trip length.
# 
# **Results**
# 
# <font color=darkgreen>Great work, and not remotely average! For the average to be a helpful summary of the data, we need for all of our durations to be reasonable numbers, and not a few that are way too big, way too small, or even malformed. For example, if there is anything fishy happening in the data, and our trip ended before it started, we'd have a negative trip length.</font>

# In[34]:


# What was the total duration of all trips?
total_elapsed_time = sum(onebike_durations)

# What was the total number of trips?
number_of_trips = len(onebike_durations)
  
# Divide the total duration by the number of trips
print(total_elapsed_time / number_of_trips)


# # <font color=darkred>2.12 The long and the short of why time is hard</font> 
# 
# Out of 291 trips taken by W20529, how long was the longest? How short was the shortest? Does anything look fishy?
# 
# As before, data has been loaded as onebike_durations.
# 
# **Instructions**
# - Calculate shortest_trip from onebike_durations.
# - Calculate longest_trip from onebike_durations.
# - Print the results, turning shortest_trip and longest_trip into strings so they can print.
# 
# **Results**
# 
# <font color=darkgreen>Weird huh?! For at least one trip, the bike returned before it left. Why could that be? Here's a hint: it happened in early November, around 2AM local time. What happens to clocks around that time each year? By the end of the next chapter, we'll have all the tools we need to deal with this situation!</font>

# In[35]:


# Calculate shortest and longest trips
shortest_trip = min(onebike_durations)
longest_trip = max(onebike_durations)

# Print out the results
print("The shortest trip was " + str(shortest_trip) + " seconds")
print("The longest trip was " + str(longest_trip) + " seconds")


# # Aditional material
# 
# - Datacamp course: https://learn.datacamp.com/courses/extreme-gradient-boosting-with-xgboost
# - Xgboost documentation: https://xgboost.readthedocs.io/en/latest/
# - sklearn.tree.DecisionTreeClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
