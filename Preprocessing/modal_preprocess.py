import os
import time

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import time
import modal
import datetime


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
        "google-cloud-storage"
    ))

stub = modal.Stub(image=image)  

@stub.function(secret=modal.Secret.from_name("my-googlecloud-secret"))
def upload_data(df, location, dir):
    from google.cloud import storage
    from google.oauth2 import service_account
    import json

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = storage.Client(credentials=credentials)

    export_bucket = client.get_bucket("location-policecall-dataset")

    #Upload dataframe to cloud storage service
    df.to_csv()

    export_bucket.blob(dir+"/"+location+".csv").upload_from_string(df.to_csv(), content_type="text/csv")

    print(dir+"/"+location+".csv", "uploaded to cloud storage")



@stub.function()
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

    #Uppercase all the column names
    df.columns = map(str.upper, df.columns)

    return df

@stub.function(timeout=10000,secret=modal.Secret.from_name("my-google-maps"))
def get_coords(row):
    import googlemaps

    #Load the API key from .env file
    gmaps = googlemaps.Client(key=os.environ["GOOGLE_MAPS"])

    geocode_result = gmaps.geocode(str(row[-3])+","+str(row[-2])+","+str(row[-1]))
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    row.append(str(lat))
    row.append(str(lng))

    time.sleep(2)

    return row


@stub.local_entrypoint()
def main():

    # record the start time
    start_time = time.time()

    distributed_df = []

    filename = "San_Francisco"
    original_df = pd.read_csv(filename+".csv")

    upload_data.call(original_df, filename, "processed-dataset")
    return
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
    for results in retain_columns.map(distributed_df):
        new_distributed_df.append(results)


    #Add all the dataframes together from distributed_df list
    df = pd.concat(new_distributed_df, ignore_index=False)
    print(df.shape)

    #Upload the dataframe to cloud storage service
    upload_data.call(df, "processed_"+filename, "processed-dataset")

    LL = ("LATITUDE", "LONGITUDE", "LAT", "LNG")

    if not any(x in df.columns for x in LL):
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
        

        final_df = pd.DataFrame(templist, columns=columns_df)
            

    print(final_df.columns)

    #Print how many rows and columns
    print(final_df.shape)

    #Display the first 5 rows
    print(final_df.head())

    #Save the dataframe to a csv file
    final_df.to_csv("processed"+filename+".csv", index=True)
    upload_data.call(final_df, "processed_"+filename, "processed-dataset")

    # record the end time
    end_time = time.time()

    # calculate the elapsed time
    elapsed_time = end_time - start_time

    # print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")

