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

    #filename = get_filename(file)

    minioClient = Minio('minio:9000',
                    access_key='breakfast',
                    secret_key='breakfast',
                    secure=False)
    # minioClient = Minio(minio_name,
    #         access_key=minio_key,
    #         secret_key=minio_secret,
    #         secure=False)
    # minioClient = Minio('127.0.0.1:9000',
    #     access_key='92WUKA7ZAP4M3UOS0TNG',
    #     secret_key='uIgJzgatEyop9ZKWfRDSlgkAhDtOzJdF+Jw+N9FE',
    #     secure=False)

    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)

    try:
           minioClient.put_object('breakfast', folder + name, f,size)

    except ResponseError as err:
           return False

    #f.save(secure_filename(f.filename))
    return {'upload':True,'location':'breakfast' + folder + '/'+ name}

# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]': return DateType()
    elif f == 'int64': return LongType()
    elif f == 'int32': return IntegerType()
    elif f == 'float64': return FloatType()
    else: return StringType()

def define_structure(string, format_type):
    try: typo = equivalent_type(format_type)
    except: typo = StringType()
    return StructField(string, typo)

def make_graphic(hr):
    time = hr['time'].to_numpy() / 60 / 60
    hr2  = hr.drop(['time','DN_RemovePoints ac1rat'],axis = 1)
    df_corr = hr2.corr()
    plt.figure(figsize=(12, 12))
    ax = sns.clustermap(
        abs(df_corr),
        cmap="YlGnBu_r"#"mako"
    )
    #ax = sns.heatmap(abs(df_corr),cmap="YlGnBu_r") #notation: "annot" not "annote"
    #bottom, top = ax.get_ylim()
    #ax.tick_params(labelsize=14)
    #ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('hist-heatmap.png',format='png', dpi=800,bbox_inches = "tight")


# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    i = 0
    for column, typo in zip(columns, types):
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)


file_location = sys.argv[2]
output_location = sys.argv[1]
job_bucket = output_location.split('/')[-1]
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print("The Job Bucket is: " + str(job_bucket))

sc =SparkContext()
sqlContext = SQLContext(sc)
#test = sc.textFile("/Users/justinniestroy-admin/Downloads/UVA_7129_HR.csv").map(lambda line: line.split(","))
test = sc.textFile(file_location).map(lambda line: line.split(","))
a = np.array(test.collect())
df = pd.DataFrame(data=a[1:,:], columns=a[0,:])
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.time = df.time.astype(float)
df_hist = df.iloc[:, : 54]
for col in df_hist.columns.tolist():
    df_hist[col] = df_hist[col].astype(float)


print(df_hist)
make_graphic(df_hist)
with open('hist-heatmap.png','rb') as f:
    upload(f,'Histogram_Heatmap.png',job_bucket + '/')
