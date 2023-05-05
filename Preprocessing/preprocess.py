import pandas as pd
import numpy as np
#Use .env file to store API key
from dotenv import load_dotenv

import googlemaps

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time
import os
import threading
import sys

#Threading with a return
class ReturnValueThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result = None

    def run(self):
        if self._target is None:
            return  # could alternatively raise an exception, depends on the use case
        try:
            self.result = self._target(*self._args, **self._kwargs)
        except Exception as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.result


#Load the API key from .env file
#gmaps = googlemaps.Client(key="""KEY""")

def retain_columns(df):
    col = ("ID", "INCIDENT", "DATE", "TIME", "LOCATION", "ADDRESS","REPORT", "TYPE", "CATEGORY", "DESCRIPTION", "CITY", "STATE", "LATITUDE","LONGITUDE")
    for c in df.columns:
        temp_c = c.upper()

        #If the string has any of the words in col, keep it
        if any(x in temp_c for x in col):
            continue
        else:
            df = df.drop(c, axis=1)
    
    #Find the column with TYPE within the name, and find all unique values
    types = list()
    colname = ""
    isFound = False
    for c in df.columns:
        typeslist = ["BURGLARY", "THEFT", "STOLEN", "SUSPICIOUS", "ASSAULT", "MISCHIEF", "BATTERY", "THREATS", "STAB", "FIGHT", "ROBBERY", "DISTURBANCE", "HATE", "KIDNAPPING"]
        #Iterate through the first 100 rows to see if they have anything from typeslist
        for i in range(0, 100):
            try:
                temp_t = df[c].iloc[i].upper()
                if any(x in temp_t for x in typeslist):
                    print("FOUND", c)
                    types = list(df[c].unique())
                    colname = c
                    isFound = True
                    break
            except:
                continue
        if isFound:
            break
    
        #Drop all rows that have a type that is not in the list
    final_types = list()
    for t in types:
        #If t is not similar to typeslist, drop it
        try:
            print("TEMP", t)
            temp_t = t.upper()
            if any(x in temp_t for x in typeslist):
                #remove from types
                final_types.append(t)
        except:
            continue
    
    #Go through the dataframe and drop all rows that have a type that is not in the list
    for index, row in df.iterrows():
        if row[colname] not in final_types:
            df = df.drop(index)
            
    #Drop all rows that have a NaN value
    df = df.dropna()

    return df


def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key="")

    geocode_result = gmaps.geocode(row['ADDRESS']+","+row['CITY']+","+row['STATE'])
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row['LAT'] = str(lat)
    row['LNG'] = str(lng)

    print(row)
    return row

def main():

    # record the start time
    start_time = time.time()

    distributed_df = []

    filename = "san_jose"
    original_df = pd.read_csv(filename+".csv")

    print(original_df.shape)

    #Split the length of the dataframe into 100 parts

    counter = 1
    for i in range(500, 0, -1):
        if len(original_df)%i == 0:
            counter = i
            break

    for i in range(0, len(original_df)-counter, counter):
        distributed_df.append(original_df[i:i+counter])
        
    new_distributed_df = []
    threads = []
    for frame in distributed_df:
        t = ReturnValueThread(target=retain_columns, args=(frame,))
        threads.append(t)
        t.start()
    
    for t in threads:
        result = t.join()
        new_distributed_df.append(result)


    #Add all the dataframes together from distributed_df list
    df = pd.concat(new_distributed_df, ignore_index=False)
    print(df.shape)


    #Save the dataframe to a csv file
    df.to_csv("processed"+filename+".csv", index=True)


    # record the end time
    end_time = time.time()

    # calculate the elapsed time
    elapsed_time = end_time - start_time

    # print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")



main()
