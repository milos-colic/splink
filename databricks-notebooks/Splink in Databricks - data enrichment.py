# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### We use the Kaggle dataset Payment Practices of UK Buyers available here https://www.kaggle.com/datasets/saikiran0684/payment-practices-of-uk-buyers
# MAGIC 
# MAGIC This dataset has been downloaded and had the column names cleaned to remove special characters.
# MAGIC 
# MAGIC Download the 

# COMMAND ----------

display(spark.read.table("robert_whiffin.splink.payment_practices_of_uk_buyers"))

# COMMAND ----------

import requests
import time
from pyspark.sql import functions as F
from pyspark.sql import Window
import pandas as pd 
from pyspark.sql import types
from pyspark.sql.functions import collect_set, size


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC follow this example t get creds
# MAGIC https://www.pythonontoast.com/python-data-interface-companies-house/

# COMMAND ----------

#key = <username from companies house api>
#
##basic = HTTPBasicAuth(key)
#companyNumber="01070807"
#api=f"https://api.company-information.service.gov.uk/company/{companyNumber}"
#requests.get(api
#             , headers={'Authorization': <auth key>}
#             
#            ).json()

# COMMAND ----------

key = <'auth key'>
def get_officers_api(companyNumber):
  api=f"https://api.company-information.service.gov.uk/company/{companyNumber}/officers"
  response = requests.get(api
               , headers={'Authorization': key}

              ).json()
  return response

# COMMAND ----------

def response_parser(response):
  items = response['items']
  output =[]
  for item in items:
    record = {}
    address = item.get('address', {})
    record['address_line_1'] = address.get('address_line_1')
    record['locality'] = address.get('locality')
    record['country'] = address.get('country')
    record['premises'] = address.get('premises')
    record['region'] = address.get('region')
    record['postal_code'] = address.get('postal_code')
    record['officer_role'] = item.get('officer_role')
    record['name'] = item.get('name')
    record['country_of_residence'] = item.get('country_of_residence')
    record['occupation'] = item.get('occupation')
    record['nationality'] = item.get('nationality')
    dob = item.get('date_of_birth')
    if dob:
      if dob.get('month') < 10:
        month = "0"+str(dob.get('month'))
      else:
        month = str(dob.get('month'))
      record['date_of_birth'] = str(dob.get('year')) + "_" + month
    else:
      record['date_of_birth'] = None
    record['officer_role'] = item.get('officer_role')
    output.append(record)

  return output

# COMMAND ----------

def get_directors(companyNumber, Report_Id):
  response=get_officers_api(companyNumber)
  parsed = response_parser(response)
  for x in parsed:  
    x["company_number"] = companyNumber
    x["Report_Id"] = Report_Id
  return parsed


# COMMAND ----------

data_to_link = spark.read.table("robert_whiffin.splink.payment_practices_of_uk_buyers")

# COMMAND ----------


data_to_enrich = (
  data_to_link
  .select("Report_Id", "company_number")
  .withColumn("row_num", F.row_number().over(Window().partitionBy().orderBy(F.lit(True))))
).cache()
n_records = data_to_enrich.count()

increment=500
i=500
enriched_results = []
while i< (n_records + increment):
  local_data = data_to_enrich.filter(
     (F.col("row_num") > (i-increment)) &
     (F.col("row_num") <= (i))
  ).toPandas()
  for row in local_data.iterrows():
    _ = row[1]
    try:
      enriched_results.append(
        get_directors(_.company_number, _.Report_Id)
      )
    except:
      pass
  
  
  spark.createDataFrame(
    pd.concat([pd.DataFrame().from_dict(x) for x in enriched_results ])
    ).write.mode("append").saveAsTable("robert_whiffin.splink.company_officers")
  print(f"enriched {len(enriched_results)} records and saved")
  print(f"{spark.sql('select count(distinct report_id) as num_records_enriched, count(*) as num_officers from robert_whiffin.splink.company_officers').collect()} records enriched so far")
  i+= increment 
  enriched_results = []
  time.sleep(60*6)
  


