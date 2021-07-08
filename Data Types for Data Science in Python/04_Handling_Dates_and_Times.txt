# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:20:49 2021

@author: jaces
"""
# Import libraries
from datetime import datetime
from datetime import timedelta

from pytz import timezone

from collections import defaultdict
from collections import Counter

from pprint import pprint

import pendulum
import pandas as pd


# Read data from file into list of list
df = pd.read_csv('data/cta_daily_summary_totals.csv', parse_dates=['service_date'])
print(df.head())

dates_list = df.resample('50d', on='service_date', label='right', offset='-1d').sum().index.strftime('%m/%d/%Y').to_list()
df['service date'] = df.service_date.dt.strftime('%m/%d/%Y')
daily_summaries = list(df[['service date', 'day_type', 'bus', 'rail_boardings', 'total_rides']].to_records(index=False))

date_ranges_list = df.resample('30d', on='service_date', label='right', offset='-1d').sum().index.strftime('%m/%d/%Y').to_list()
date_ranges_str = list(zip(date_ranges_list[0::2], date_ranges_list[1::2]))
date_ranges_list = [datetime.strptime(d,'%m/%d/%Y') for d in date_ranges_list]
date_ranges = list(zip(date_ranges_list[0::2], date_ranges_list[1::2]))

df_rail = df.sort_values(by='service_date').groupby(['service_date', 'service date']).rail_boardings.sum().reset_index()
NY_ridership = df_rail[['service date', 'rail_boardings']].to_records(index = False)

#df_rail['service_date'].apply(lambda x: datetime(x.year, x.month, x.day))
#NY_ridership = df_rail.to_records(index=False)
# Get the year of a np.datetime
#print(NY_ridership[0][0])
#print(NY_ridership[0][0].astype('datetime64[Y]').astype(int) + 1970)

df = pd.read_csv('data/crime_sampler.csv', parse_dates=['Date'])
print(df.head())

parking_violations_dates = df[df['Location Description'].str.contains('PARKING', na=False, regex=False
                                                                     )].Date.dt.strftime('%m/%d/%Y').to_list()

df['day'] = df.Date.dt.strftime('%Y-%m-%d')
df['time'] = df.Date.dt.strftime('%H:%M:%S')
parking_violations = list(df[df['Location Description'].str.contains('PARKING', na=False,
                                                                     regex=False)][['day','time']].to_records(index = False))


print('*********************************************************')
print('BEGIN')
print('*********************************************************')
print('** 04 Handling Dates and Times')
print('*********************************************************')
print('** 04.01 There and Back Again a DateTime Journey')
print('*********************************************************')
# From string to datetime
parking_violations_date = '06/11/2016'

# Parsing strings into datetimes
date_dt = datetime.strptime(parking_violations_date, '%m/%d/%Y')
print(type(date_dt), date_dt)

# From Datetime to String
print(date_dt.strftime('%m/%d/%Y'))

# Printing a datetime as an ISO standard string
print(date_dt.isoformat())

print('*********************************************************')
print('** 04.02 Strings to DateTimes')
print('*********************************************************')
# Iterate over the dates_list 
for date_str in dates_list[:10]:
    # Convert each date to a datetime object: date_dt
    date_dt = datetime.strptime(date_str, '%m/%d/%Y')
    
    # Print each date_dt
    print(date_dt)
    
datetimes_list = []

# Iterate over the dates_list 
for date_str in dates_list:
    # Convert each date to a datetime object: date_dt
    date_dt = datetime.strptime(date_str, '%m/%d/%Y')
    
    datetimes_list.append(date_dt)

# Print first 10 items of datetimes_list
pprint(datetimes_list[:10])
    
print('*********************************************************')
print('** 04.03 Converting to a String')
print('*********************************************************')
# Loop over the first 10 items of the datetimes_list
for item in datetimes_list[:10]:
    # Print out the record as a string in the format of 'MM/DD/YYYY'
    print(item.strftime('%m/%d/%Y'))
    
    # Print out the record as an ISO standard string
    print(item.isoformat())
    
print('*********************************************************')
print('** 04.04 Working with Datetime Components and current time')
print('*********************************************************')
print(parking_violations_dates[:4])
daily_violations = defaultdict(int)

for violation in parking_violations_dates:
    violation_date = datetime.strptime(violation, '%m/%d/%Y')
    daily_violations[violation_date.day] += 1

print(daily_violations)

print('*********************************************************')
print('** 04.05 Pieces of Time')
print('*********************************************************')
# Create a defaultdict of an integer: monthly_total_rides
monthly_total_rides = defaultdict(int)

# Loop over the list daily_summaries
for daily_summary in daily_summaries:
    # Convert the service_date to a datetime object
    service_datetime = datetime.strptime(daily_summary[0], '%m/%d/%Y')

    # Add the total rides to the current amount for the month
    monthly_total_rides[service_datetime.month] += int(daily_summary[4])
    
# Print monthly_total_rides
pprint(monthly_total_rides)

# Create a Counter: monthly_total_rides
monthly_total_rides = Counter([datetime.strptime(daily_summary[0], '%m/%d/%Y').month for daily_summary in daily_summaries])
print(monthly_total_rides)
print(monthly_total_rides.most_common(1))

print('*********************************************************')
print('** 04.06 Creating DateTime Objects... Now')
print('*********************************************************')
# Compute the local datetime: local_dt
local_dt = datetime.now()

# Print the local datetime
print(local_dt)

# Compute the UTC datetime: utc_dt
utc_dt = datetime.utcnow()

# Print the UTC datetime
print(utc_dt)

print('*********************************************************')
print('** 04.07 Timezones')
print('*********************************************************')
# Print total_rides
pprint(NY_ridership[:2])

# Create a defaultdict of an integer: total_rides
total_rides = []

# Loop over the list daily_summaries
for daily_summary in NY_ridership:
    # Add the total rides to the current amount for date
    total_rides.append((datetime.strptime(daily_summary[0], '%m/%d/%Y'), daily_summary[1]))
    
# Print total_rides
pprint(total_rides[:2])

# Create a Timezone object for Chicago
chicago_usa_tz = timezone('US/Central')

# Create a Timezone object for New York
ny_usa_tz = timezone('US/Eastern')

for orig_dt, ridership in total_rides[:5]:

    # Make the orig_dt timezone "aware" for Chicago
    chicago_dt = orig_dt.replace(tzinfo=chicago_usa_tz)
    
    # Convert chicago_dt to the New York Timezone
    ny_dt = chicago_dt.astimezone(ny_usa_tz)
    
    # Print the chicago_dt, ny_dt, and ridership
    print('Chicago: %s, NY: %s, Ridership: %s' % (chicago_dt, ny_dt, ridership))
    
print('*********************************************************')
print('** 04.08 Time Travel (Adding and Subtracting Time)')
print('*********************************************************')
flashback = timedelta(days=90)
print('Date 1:', orig_dt)
print('Date 2:', local_dt)

# Subtracting timedeltas
print('Data 1 - 90 días:', orig_dt - flashback)

# Adding timedeltas
print('Data 1 + 90 días', orig_dt + flashback)

# Datetime differences
time_diff = local_dt - orig_dt
print('Difference between dates:', type(time_diff))
print(time_diff)

print('*********************************************************')
print('** 02.09 Finding a time in the future and from the past')
print('*********************************************************')
# Create the variables
review_dates = [datetime(2013, 12, 22, 0, 0),
                datetime(2013, 12, 23, 0, 0),
                datetime(2013, 12, 24, 0, 0),
                datetime(2013, 12, 25, 0, 0),
                datetime(2013, 12, 26, 0, 0),
                datetime(2013, 12, 27, 0, 0),
                datetime(2013, 12, 28, 0, 0),
                datetime(2013, 12, 29, 0, 0),
                datetime(2013, 12, 30, 0, 0),
                datetime(2013, 12, 31, 0, 0)]

new_ds = {datetime.strptime(d, '%m/%d/%Y'): {'day_type': dt, 'total_ridership': tr} for d, dt, _, _, tr in daily_summaries}

# Build a timedelta of 30 days: glanceback
glanceback = timedelta(days=30)

# Iterate over the review_dates as date
for date in review_dates:
    # Calculate the date 30 days back: prior_period_dt
    prior_period_dt = date - glanceback
    
    # Print the review_date, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
         (date, 
          new_ds[date]['day_type'], 
          new_ds[date]['total_ridership']))

    # Print the prior_period_dt, day_type and total_ridership
    print('Date: %s, Type: %s, Total Ridership: %s' %
         (prior_period_dt, 
          new_ds[prior_period_dt]['day_type'], 
          new_ds[prior_period_dt]['total_ridership']))
    
print('*********************************************************')
print('** 02.10 Finding differences in DateTimes')
print('*********************************************************')
# Iterate over the date_ranges
for start_date, end_date in date_ranges[:10]:
    # Print the End and Start Date
    print(end_date, start_date)
    # Print the difference between each end and start date
    print(end_date - start_date)
    
print('*********************************************************')
print('** 04.11 HELP! Libraries to make it easier')
print('*********************************************************')
# Parsing time with pendulum
occurred = parking_violations[0][0] + ' ' + parking_violations[0][1]
print('In:', occurred)

occurred_dt = pendulum.parse(occurred, tz='US/Eastern')
print('Pendulum type (US/Eastern):', occurred_dt)

# Timezone hopping with pendulum
print('Pendulum type (US/Central):', occurred_dt.in_timezone('US/Central'))
print('Pendulum type (Asia/Tokyo):', occurred_dt.in_timezone('Asia/Tokyo'))
print('Now             :', pendulum.now()) #always be UTC
print('Now (US/Eastern):', pendulum.now('US/Eastern'))
print('Now (Asia/Tokyo):', pendulum.now('Asia/Tokyo')) 

# Humanizing differences
now_day = pendulum.now()
diff = now_day - occurred_dt
print(diff)
print(diff.in_years(), 'years')
print(diff.in_months(), 'months')
print(diff.in_days(), 'days')
print(diff.in_hours(), 'hours')
print(diff.in_words())

# Set spanish
pendulum.set_locale('es')
print(diff.in_words())

print('*********************************************************')
print('** 04.12 Localizing time with pendulum')
print('*********************************************************')
# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')
print(tokyo_dt)
print(tokyo_dt.to_iso8601_string())

# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')

# Print the ISO 8601 string of la_dt
print(la_dt)
print(la_dt.to_iso8601_string())

print('*********************************************************')
print('** 04.13 Humanizing Differences with Pendulum')
print('*********************************************************')
# Iterate over date_ranges
for start_date, end_date in date_ranges_str[:5]:

    # Convert the start_date string to a pendulum date: start_dt 
    start_dt = pendulum.parse(start_date, strict = False)
    
    # Convert the end_date string to a pendulum date: end_dt 
    end_dt = pendulum.parse(end_date, strict = False)
    
    # Print the End and Start Date
    print(end_dt, start_dt)
    
    # Calculate the difference between end_dt and start_dt: diff_period
    diff_period = end_dt - start_dt
    
    # Print the difference in days
    print(diff_period.in_days())
    
print('*********************************************************')
print('END')
print('*********************************************************')