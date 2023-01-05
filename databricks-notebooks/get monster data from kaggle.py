# Databricks notebook source
pip install kaggle

# COMMAND ----------

download_path = "dbfs:/Users/robert.whiffin@databricks.com/datasets/kaggle/job-listing-dataset-monster-uk"
dbutils.fs.mkdirs(download_path)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC #rm ~/.kaggle/kaggle.json
# MAGIC mkdir ~/.kaggle
# MAGIC echo '{"username":<kaggle username>,"key":<kaggle key>}' > ~/.kaggle/kaggle.json

# COMMAND ----------

# MAGIC %sh 
# MAGIC kaggle datasets download --force -d promptcloud/job-listing-dataset-monster-uk -p /dbfs/Users/robert.whiffin@databricks.com/datasets/kaggle/job-listing-dataset-monster-uk

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cd /dbfs/Users/robert.whiffin@databricks.com/datasets/kaggle/job-listing-dataset-monster-uk
# MAGIC unzip job-listing-dataset-monster-uk.zip

# COMMAND ----------



# COMMAND ----------

(
spark.read.format('json')
.load("dbfs:/Users/robert.whiffin@databricks.com/datasets/kaggle/job-listing-dataset-monster-uk/marketing_sample_for_monster_uk-monster_uk_job__20201201_20210331__30k_data.ldjson")
.write
.saveAsTable("robert_whiffin.splink.monster_job_descriptions")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from robert_whiffin.splink.monster_job_descriptions
