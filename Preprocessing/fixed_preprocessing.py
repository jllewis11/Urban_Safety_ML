import pandas
import datetime

pd_old = pandas.read_csv("Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv")

print(pd_old.head())

print(pd_old.columns)
print(pd_old.shape)

#Keep Category, Date, Time, X, Y columns
pd_new = pd_old[['Category', 'Date', 'Time', 'X', 'Y']]
print(pd_new.head())
print(pd_new.shape)

#Print all unique values in Category column
print(pd_new['Category'].unique())

columns = ['ROBBERY', 'VEHICLE THEFT', 'ASSAULT', 'BURGLARY', 'LARCENY/THEFT', ' KIDNAPPING', 'STOLEN PROPERTY']
#Change LARCENY/THEFT to THEFT
pd_new['Category'] = pd_new['Category'].apply(lambda x: x.replace('LARCENY/THEFT', 'THEFT'))

#Filter out all rows that are not in columns
pd_new = pd_new[pd_new['Category'].isin(columns)]
print(pd_new.head())
print(pd_new.shape)


pd2_old = pandas.read_csv("Police_Department_Incident_Reports__2018_to_Present.csv")
print(pd2_old.head())
print(pd2_old.columns)
print(pd2_old.shape)

print(pd2_old['Incident Category'].unique())
#keep Incident Datetime, Incident Category, Latitutde, Longitude
pd2_old = pd2_old[['Incident Datetime', 'Incident Category', 'Latitude', 'Longitude']]

#ensure that all values in Incident category are strings
pd2_old['Incident Category'] = pd2_old['Incident Category'].astype(str)

columns =['Larceny Theft', 'Assault', 'Robbery', 'Stolen Property', 'Motor Vehicle Theft', 'Burglary', 'Motor Vehicle Theft?', 'Human Trafficking (A), Commercial Sex Acts', 'Human Trafficking, Commercial Sex Acts', 'Human Trafficking (B), Involuntary Servitude']
#Change 'Human Trafficking (A), Commercial Sex Acts', 'Human Trafficking, Commercial Sex Acts' 'Weapons Offence', 'Human Trafficking (B), Involuntary Servitude' to KIDNAPPING
pd2_old['Incident Category'] = pd2_old['Incident Category'].apply(lambda x: x.replace('Human Trafficking (A), Commercial Sex Acts', 'KIDNAPPING'))
pd2_old['Incident Category'] = pd2_old['Incident Category'].apply(lambda x: x.replace('Human Trafficking, Commercial Sex Acts', 'KIDNAPPING'))
pd2_old['Incident Category'] = pd2_old['Incident Category'].apply(lambda x: x.replace('Human Trafficking (B), Involuntary Servitude', 'KIDNAPPING'))
pd2_old['Incident Category'] = pd2_old['Incident Category'].apply(lambda x: x.replace('Motor Vehicle Theft?', 'Motor Vehicle Theft'))

#Filter out all rows that are not in columns
pd2_new = pd2_old[pd2_old['Incident Category'].isin(columns)]
#Drop any rows with NaN values
pd2_new = pd2_new.dropna()
print(pd2_new.head())
print(pd2_new.shape)


#Combine Column date and time into one column called datetime using datetime
#datetime.strptime(time_string, "%m/%d/%Y %H:%M")

#Convert Date and Time to datetime
#2004-11-22 17:50:00
pd_new['Date'] = pd_new['Date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
pd_new['Time'] = pd_new['Time'].apply(lambda x: datetime.datetime.strptime(x, "%H:%M"))

#Combine Date and Time into one column
pd_new['Datetime'] = pd_new['Date'] + pd_new['Time'].dt.time.apply(lambda t: pandas.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second))

#Replace column Y with latitude
#Replace column X with longitude
pd_new = pd_new.rename(columns={'Y': 'Latitude', 'X': 'Longitude'})

#Reorder the columns by Datetime, Category, latitude, longitude
pd_new = pd_new[['Datetime', 'Category', 'Latitude', 'Longitude']]

print(pd_new.head())

#REplace column Incident Datetime with Datetime, Incident Category with Category
pd2_new = pd2_new.rename(columns={'Incident Datetime': 'Datetime', 'Incident Category': 'Category'})

#Replace Incident Datetime parsed_time = datetime.strptime(time_string, "%Y/%m/%d %I:%M:%S %p")
pd2_new['Datetime'] = pd2_new['Datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %I:%M:%S %p"))

#Replace Incident Category with upper for each word
pd2_new['Category'] = pd2_new['Category'].apply(lambda x: x.upper())

print(pd2_new.head())


#Combine pd_new and pd2_new
pd_new = pd_new.append(pd2_new, ignore_index=True)
#Seprate Date and Time into two columns
pd_new['Date'] = pd_new['Datetime'].dt.date
pd_new['Time'] = pd_new['Datetime'].dt.time

#Drop Datetime column
pd_new = pd_new.drop(columns=['Datetime'])

def categorize_part_of_day(time_obj):
    hour = time_obj.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

# Assuming you have a DataFrame called 'data' with a column named 'Time'
pd_new['Part_of_Day'] = pd_new['Time'].apply(categorize_part_of_day)

def get_day_of_week(date_str):
    date_obj = pandas.to_datetime(date_str)
    return date_obj.day_name()

# Assuming you have a DataFrame called 'data' with a column named 'Date'
pd_new['Day_of_Week'] = pd_new['Date'].apply(get_day_of_week)


#Reorder the columns by Date, Time, Category, latitude, longitude
pd_new = pd_new[['Date', 'Time', 'Day_of_Week', 'Part_of_Day', 'Category', 'Latitude', 'Longitude']]



print(pd_new.head())
print(pd_new.shape)

print(pd_new['Category'].unique())

#Save to csv
pd_new.to_csv("San_Francisco.csv", index=False)
