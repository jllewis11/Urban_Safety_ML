import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time
import modal


image=(modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "apt-get update",
    ).pip_install(
        "pandas",
        "scikit-learn",
        "numpy",
        "requests",
        "googlemaps",
    ))

stub = modal.Stub(image=image)  

@stub.function(timeout=86400)
def retain_columns(df):
    col = ("ID", "INCIDENT", "DATE", "TIME", "LOCATION", "ADDRESS","TYPE", "CATEGORY", "DESCRIPTION", "CITY", "STATE")
    for c in df.columns:
        c = c.upper()
        #If the string has any of the words in col, keep it
        if any(x in c for x in col):
            continue
        else:
            df = df.drop(c, axis=1)
    #Find the column with TYPE within the name, and find all unique values
    types = list()
    colname = ""
    for c in df.columns:
        c = c.upper()
        if "TYPE" in c and "CODE" not in c:
            #print(df[c].unique())
            types = list(df[c].unique())
            colname = c

    #Drop all rows that have a type that is not in the list
    typeslist = ["BURGLARY", "DISTURBANCE", "THEFT", "STOLEN", "SUSPICIOUS", "ASSAULT", "MISCHIEF", "BATTERY", "THREATS", "STAB", "FIGHT", "ROBBERY", "DISTURBANCE", "HATE", "KIDNAPPING"]
    for t in types:
        #If t is not similar to typeslist, drop it
        if not any(x in t for x in typeslist):
            #remove from types
            types.remove(t)
    
    #Go through the dataframe and drop all rows that have a type that is not in the list
    for index, row in df.iterrows():
        if row[colname] not in types:
            df = df.drop(index)
            print(index)

    return df

@stub.function(timeout=86400,secret=modal.Secret.from_name("my-google-maps"))
def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key=os.environ["GOOGLE_MAPS"])

    geocode_result = gmaps.geocode(str(row[-3])+","+str(row[-2])+","+str(row[-1]))
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row.append(str(lat))
    row.append(str(lng))

    return row


@stub.local_entrypoint()
def main():
    df = pd.read_csv("sanjose_policecalls2023.csv")

    
    df = retain_columns.call(df)
    df.to_csv("preprocessed_sanjose_policecalls2023.csv", index=True)

    #Convert df to list
    list_df = df.values.tolist()

    # #Adding the new columns
    columns_df = df.columns.tolist()
    columns_df.append("LAT")
    columns_df.append("LNG")

    templist = list()

    #Find the biggest whole number that divides evenly, but is smaller or equal to 50 for the number len(list_sanjose)
    #This is to avoid overloading the API
    counter = 1
    for i in range(50, 0, -1):
        if len(list_df)%i == 0:
            counter = i
            break

    for x in range(0, len(list_df)-counter, counter):
        templist += list(get_coords.map(list_df[x:x+counter]))
        print(len(templist))
        time.sleep(1)

    
    #print(len(templist[0]))
    time.sleep(3)

    df = pd.DataFrame(templist, columns=columns_df)
        

    print(df.columns)

    #Print how many rows and columns
    print(df.shape)

    #Display the first 5 rows
    print(df.head())

    #Save the dataframe to a csv file
    df.to_csv("updated_sanjose_policecalls2023.csv", index=True)


