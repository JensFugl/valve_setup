# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:09:11 2020

@author: Jens Ringsholm
"""

import boto3
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timezone


from creds import access_key_id, secret_access_key

plt.close('all')

########################### functions ###########################################

def CSVwriter(filename, data, col_names):

    with open(filename, 'w') as csvFile:
        writer = csv.DictWriter(
        csvFile, fieldnames=col_names)
        writer.writeheader()
        writer = csv.writer(csvFile, delimiter=',', quotechar='|')
        for row in data:
            writer.writerows(data)
            csvFile.flush()
    csvFile.close()

####################### Import Stuff  ##########################################

time1 = time.strptime('2020-05-31 16:50:00', '%Y-%m-%d %H:%M:%S')
time2 = time.strptime('2020-06-01 21:00:00', '%Y-%m-%d %H:%M:%S')

plot = True
write = True



##########################  Look no furter #####################################



myNode4 = "BN1000"


epoch1 = time.mktime(time1)
epoch2 = time.mktime(time2)
batchsize = 60*60
epoch_list = np.arange(epoch1, epoch2, batchsize).tolist()


client = boto3.client(
    'dynamodb',
    # Hard coded strings as credentials, not recommended (pythonScript).
    aws_access_key_id= access_key_id, 
    aws_secret_access_key=secret_access_key,
    region_name='eu-central-1'
)


payload4 = []
for ep in epoch_list:
    response4 = client.query(
        TableName='data_db',
        KeyConditionExpression="#S = :deviceId AND #T BETWEEN :ep1 AND :ep2",
        ExpressionAttributeNames={
            "#S": "devId", "#T": "epoch"
        },
        ExpressionAttributeValues={
            ":deviceId": {"S": myNode4},
            ":ep1": {"N": str(int(ep))},
            ":ep2": {"N": str(int(ep+batchsize))}
        }
    )
    for itm in response4['Items']:
        payload4.append(itm)


epoch=[]
rt=[]
pres=[]
temp = []
for itm in payload4:
    
    
    for i in range(len(itm['payload']['M']['MS5803']['M']['T']['L'])):       
        pres.append(float(itm['payload']['M']['MS5803']['M']['P']['L'][i]['N']))
        temp.append(float(itm['payload']['M']['MS5803']['M']['T']['L'][i]['N']))   
        
        epoch.append(int(itm['epoch']['N'])-10+(2*i))
        rt.append(datetime.fromtimestamp(epoch[-1], timezone.utc))
   
data = (zip(epoch, pres, temp, rt))


    ########################CSV writer 8########################

if write == True:
    CSVwriter('data/valve_data.csv', data, ['Time', 'Pressure', 'Temperature', 'rt'])



epoch_4 = np.array(epoch)

epoch_4 = (epoch_4-max(epoch_4))/60


fig1, ax1 = plt.subplots(figsize=(12, 6))         
plt.plot(rt, pres)
plt.show()
 

fig1, ax1 = plt.subplots(figsize=(12, 6))         
plt.plot(rt, temp)
plt.show()
  

fig1, ax1 = plt.subplots(figsize=(12, 6))         
plt.plot(epoch)
plt.show()

fig1, ax1 = plt.subplots(figsize=(12, 6))         
plt.plot(epoch, rt)
plt.show()
