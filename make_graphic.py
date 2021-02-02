from pyspark import SparkContext
import sys
import os
import seaborn as sns
import subprocess
from minio import Minio
import numpy as np
import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import SQLContext
import matplotlib.pyplot as plt



def upload(f,name,folder):

    minioClient = Minio('minio:9000',
                    access_key='breakfast',
                    secret_key='breakfast',
                    secure=False)

    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    try:
           minioClient.put_object('breakfast', folder + name, f,size)

    except ResponseError as err:
           return False

    return {'upload':True,'location':'breakfast' + folder + '/'+ name}

def make_graphic(hr):
    df_corr = hr.corr()
    plt.figure(figsize=(12, 12))
    ax = sns.clustermap(
        abs(df_corr),
        cmap="YlGnBu_r",#"mako",
        xticklabels=False,
        yticklabels=False
    )
    ax = ax.ax_heatmap
    ax.set_xlabel("Correlation of Algorithms",fontsize=20)
    plt.savefig('hist-heatmap.png',format='png', dpi=800,bbox_inches = "tight")




file_location = sys.argv[2]
output_location = sys.argv[1]
job_bucket = output_location.split('/')[-1]
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print("The Job Bucket is: " + str(job_bucket))

sc =SparkContext()
sqlContext = SQLContext(sc)

test = sc.textFile(file_location).map(lambda line: line.split(","))
a = np.array(test.collect())

df = pd.DataFrame(data=a[1:,:], columns=a[0,:])
df_hist = df.iloc[:,:-1]
df_hist  = df_hist.drop(['IsSeasonal?'],axis = 1)

for col in df_hist.columns.tolist():
    try:
        df_hist[col] = df_hist[col].astype(float)
    except:
        df_hist = df_hist.drop([col],axis = 1)

df_hist = df_hist.loc[:, df_hist.std() > 0]
df_hist = df_hist[np.isfinite(df_hist).all(1)]


print(df_hist)
make_graphic(df_hist)

with open('hist-heatmap.png','rb') as f:
    upload(f,'HCTSA Heatmap.png',job_bucket + '/')
