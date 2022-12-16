# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### We use the Kaggle dataset Payment Practices of UK Buyers available here https://www.kaggle.com/datasets/saikiran0684/payment-practices-of-uk-buyers
# MAGIC 
# MAGIC This dataset has been downloaded and had the column names cleaned to remove special characters

# COMMAND ----------

pip install git+https://github.com/robertwhiffin/splink.git@mllfow-integration

# COMMAND ----------

import requests
import time
from pyspark.sql import functions as F
from pyspark.sql import Window
import splink.spark.spark_comparison_library as cl
import pandas as pd 
from pyspark.sql import types
from pyspark.sql.functions import collect_set, size
from splink.mlflow import splink_mlflow
import mlflow

# COMMAND ----------

#display(
#spark.read.table("robert_whiffin.splink.payment_practices_of_uk_buyers")
#)
#display(
#spark.read.table("robert_whiffin.splink.company_officers")
#)

# COMMAND ----------

#companies = spark.read.table("robert_whiffin.splink.payment_practices_of_uk_buyers")
#officers = spark.read.table("robert_whiffin.splink.company_officers")
#
#joined_data = (
#  companies
#  .join(officers, ["Report_Id", "company_number"], "left")
#  .withColumn("forenames", F.split("name", ",")[1])
#  .withColumn("surname", F.split("name", ",")[0])
#  .select(
#    "name"
#    ,"country_of_residence"
#    ,"nationality"
#    ,"date_of_birth"
#    ,"forenames"
#    ,"surname"
#    ,"postal_code"
#    ,"premises"
#    ,"address_line_1"
#    ,"locality"
#    ,"region"
#  )
#.drop_duplicates()
#  .withColumn("uid", F.monotonically_increasing_id())
#)
#
#joined_data.write.option("overwriteSchema", "true").mode("overwrite").saveAsTable("robert_whiffin.splink.companies_with_officers")
joined_data = (
  spark.read.table("robert_whiffin.splink.companies_with_officers")
)  
#display(joined_data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Comparison Library - comparison function types
# MAGIC 
# MAGIC https://moj-analytical-services.github.io/splink/comparison_library.html
# MAGIC 
# MAGIC 
# MAGIC - exact_match()
# MAGIC   - Columns values must be identical
# MAGIC   - Useful for...
# MAGIC   
# MAGIC - distance_function_at_thresholds()
# MAGIC   - Provide a user defined distance function and threshold. By default will include exact matches too.
# MAGIC   
# MAGIC   
# MAGIC - levenshtein_at_thresholds()
# MAGIC   - Comparison is made using Levenshtein distance (number of characters different in string, i.e. LD(test, tent) = 1, LD(one, two) = 3)
# MAGIC   - Useful for...
# MAGIC   
# MAGIC - jaccard_at_thresholds()
# MAGIC   - Comparison is made using Jaccard similarity (set difference, i.e JS (one, two) = 2, JD(xyz, zxy) = 0).
# MAGIC   - Useful for...
# MAGIC   
# MAGIC - jaro_winkler_at_thresholds()
# MAGIC   - jaro winkler similarity looks at entire string similarity by considering the relative position of each character in the string - characters are considered a match if they are the same character and within 2 characters of the same location in the comparison string. 
# MAGIC   - Useful for...
# MAGIC   
# MAGIC Examples of comparisons under different metrics (I'm not sure these are correct)
# MAGIC 
# MAGIC | Comparison            | Exact Match | Levenshtein | Jaccard | Jaro Winkler |
# MAGIC |-----------------------|-------------|-------------|---------|--------------|
# MAGIC | farmer, famrer        | 0           | 1           | 1       | 0.95         |
# MAGIC | clean, cleaner        | 0           | 2           | 0.83    | 0.94         |
# MAGIC | Birmingham, Birginham | 0           | 2           | 1       | 0.93         |

# COMMAND ----------

# This is used to reduce the potential pair space to a tractable number
# Blocking rules are defined as SQL expressions
blocking_rules_to_generate_predictions = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.nationality = r.nationality and l.locality = r.locality and l.premises = r.premises and l.region = r.region",
  "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code and l.surname = r.surname",
  
]
# define how different attributes between record pairs will be comapr
comparisons = [
  cl.jaro_winkler_at_thresholds("name", 0.6),
  cl.jaro_winkler_at_thresholds("surname", 0.8),
  cl.jaro_winkler_at_thresholds("forenames", 0.7),
  cl.jaro_winkler_at_thresholds("address_line_1", 0.9),
  cl.levenshtein_at_thresholds("country_of_residence", 0.7),
  cl.levenshtein_at_thresholds("nationality", 0.7),
]

settings = {
    "link_type": "dedupe_only",
    "comparisons": comparisons,
    "blocking_rules_to_generate_predictions": blocking_rules_to_generate_predictions,
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    "em_convergence": 0.01,
    "unique_id_column_name": "uid"
}

# COMMAND ----------

# MAGIC 
# MAGIC %sql
# MAGIC 
# MAGIC --create catalog robert_whiffin;
# MAGIC use catalog robert_whiffin;
# MAGIC --create database splink;
# MAGIC use database splink;
# MAGIC 
# MAGIC show tables

# COMMAND ----------

# Seems like this is necessary to drop tables in case of changing the underlying data - must be an easier way of presseing the reset button?
#splink_tables = spark.sql('show tables like "*__splink__*"')
#
#temp_tables = splink_tables.collect()
#drop_tables = list(map(lambda x: x.tableName, temp_tables))
#for x in drop_tables:
#    spark.sql(f"drop table {x}")

# COMMAND ----------

from splink.spark.spark_linker import SparkLinker
linker = SparkLinker(joined_data, database = "splink", catalog = "robert_whiffin", settings_dict=settings, spark=spark)

# COMMAND ----------

linker.profile_columns("country_of_residence ")

# COMMAND ----------

# estimate how many record pairs we will generate

for rule in blocking_rules_to_generate_predictions:
  prediction = linker.count_num_comparisons_from_blocking_rule(rule)
  print(f"Number of comparisons generated by rule '{rule}': {prediction}")

# COMMAND ----------


deterministic_rules = [
  "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
  "l.address_line_1 = r.address_line_1 and jaro_winkler(l.name, r.name) <= 5",
  "l.name = r.name and jaro_winkler(l.address_line_1, r.address_line_1) <= 5",
]

# COMMAND ----------

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.7)

# COMMAND ----------

linker.estimate_u_using_random_sampling(target_rows=1e6)


# COMMAND ----------

training_rules = [
  "l.name = r.name and l.date_of_birth = r.date_of_birth",
  "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code",
]

# COMMAND ----------

training_results = []
for TR in training_rules:
  training_results.append(linker.estimate_parameters_using_expectation_maximisation(TR))


# COMMAND ----------

experiment_name = "/Users/robert.whiffin@databricks.com/Splink/experiment2"
experiment = mlflow.set_experiment(experiment_name=experiment_name)

model_name="linker"
run = splink_mlflow.log_splink_model_to_mlflow(linker, model_name)

# COMMAND ----------

register=mlflow.register_model(
  f"runs:/{run.info.run_id}/{model_name}"
  , name='splink_linker2'
)

# COMMAND ----------

register

# COMMAND ----------

model_uri = f"models:/{register.name}/{register.version}"
json = splink_mlflow.get_model_json(model_uri)

# COMMAND ----------

params={"test1":"test1", "test2":"test2"}
metrics={"metric1":1, "metric2":2}

run = splink_mlflow.log_splink_model_to_mlflow(linker, model_name, params=params, metrics=metrics)

# COMMAND ----------

linker.match_weights_chart()

# COMMAND ----------

linker.m_u_parameters_chart()

# COMMAND ----------

linker.unlinkables_chart()

# COMMAND ----------

results = linker.predict(threshold_match_probability=0.7)

# COMMAND ----------

results.as_pandas_dataframe(limit=20)

# COMMAND ----------

records_to_view  = results.as_record_dict(limit = 1000)
linker.waterfall_chart(records_to_view, filter_nulls=False)

# COMMAND ----------

#clusters = linker.cluster_pairwise_predictions_at_threshold(results, threshold_match_probability=0.7)
clusters.as_pandas_dataframe(limit=20)

# COMMAND ----------

display(
  clusters.as_spark_dataframe()
  .orderBy("cluster_id")
)

# COMMAND ----------

import mlflow

# COMMAND ----------

experiment_name = "/Users/robert.whiffin@databricks.com/Splink/experiment1"

#experiment_id = mlflow.create_experiment(experiment_name)
experiment = mlflow.set_experiment(experiment_name=experiment_name)

# COMMAND ----------

# things to log - mostly need extracting from the settings dict
splink_model = linker._settings_obj.as_dict()
comparisons = splink_model['comparisons']

  
#======================================================================
#======================================================================
def log_comparison_details(comparison):
  output_column_name = comparison['output_column_name']
  
  comparison_levels = comparison['comparison_levels']
  for _ in comparison_levels:
    sql_condition = _.get('sql_condition')
    m_probability = _.get('m_probability')
    u_probability = _.get('u_probability')
    log_dict = {
      f"output column {output_column_name} compared through condition {sql_condition} m_probability": m_probability,
      f"output column {output_column_name} compared through condition {sql_condition} u_probability": u_probability
      
    }
    if m_probability:
      mlflow.log_params(log_dict)
  
#======================================================================
#======================================================================
def log_comparisons(splink_model):
  comparisons = splink_model['comparisons']
  for _ in comparisons:
    log_comparison_details(_)
  
#======================================================================
#======================================================================
def log_hyperparameters(splink_model):
  hyper_param_keys = ['link_type', 'probability_two_random_records_match', 'sql_dialect', 'unique_id_column_name', 'em_convergence', 'retain_matching_columns', 'blocking_rules_to_generate_predictions']
  for key in hyper_param_keys:
    mlflow.log_param(key, splink_model[key])
  
#======================================================================
#======================================================================
def log_splink_model_json(splink_model):
  mlflow.log_dict(splink_model, "linker.json")
  
#======================================================================
#======================================================================
class splinkMlflow(mlflow.pyfunc.PythonModel):
  def __init__(self, linker_json):
    self.linker_json = linker_json
    
  def predict(self, context):
    return self.json
  
  def model_json(self):
    return self.json
  
#======================================================================
#======================================================================  
class splinkSparkMLFlowWrapper(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    self.linker_json = context.artifacts['linker_json_path']
    
  def model_json(self):
    return self.json
#======================================================================
#======================================================================
def save_splink_model_to_mlflow(linker, model_name):
  #mlflowSplink = splinkMlflow(linker)
  #mlflow.pyfunc.log_model(model_name, python_model=mlflowSplink)
  path="linker.json"
  if os.path.isfile(path):
    os.remove(path)
  linker.save_settings_to_json(path)
  artifacts = {"linker_json_path": path}
  mlflow.pyfunc.log_model(model_name, python_model=splinkSparkMLFlowWrapper(), artifacts = artifacts)
  os.remove(path)
#======================================================================
#======================================================================  
def log_splink_model_to_mlflow(linker,model_name="linker", log_charts=True):
  splink_model_json = linker._settings_obj.as_dict()
  
  with mlflow.start_run() as run:
    log_splink_model_json(splink_model_json)
    log_hyperparameters(splink_model_json)
    log_comparisons(splink_model_json)
    save_splink_model_to_mlflow(linker, model_name)
    if log_charts:
      log_linker_charts(linker)
      
  return run
  
#======================================================================
#======================================================================  
def log_linker_charts(linker):
  weights_chart = linker.match_weights_chart()
  mu_chart = linker.m_u_parameters_chart()
  missingness_chart = linker.missingness_chart()
  #completeness = linker.completeness_chart()
  compare_chart = linker.parameter_estimate_comparisons_chart()
  unlinkables = linker.unlinkables_chart()
  blocking_rules_chart = linker.cumulative_num_comparisons_from_blocking_rules_chart()
  charts_dict = {
    "weights_chart": weights_chart,
    "mu_chart": mu_chart,
    "missingness_chart": missingness_chart,
  #  "completeness": completeness,
    "compare_chart": compare_chart,
    "unlinkables": unlinkables,
    "blocking_rules_chart": blocking_rules_chart,
  }
  for name, chart in charts_dict.items():
    log_chart(name, chart)
  
#======================================================================
#======================================================================  
def log_chart(chart_name, chart):
  path=f"{chart_name}.html"
  if os.path.isfile(path):
    os.remove(path)
  save_offline_chart(chart.spec, path)
  mlflow.log_artifact(path)

# COMMAND ----------

run = log_splink_model_to_mlflow(linker, "linker", False)

# COMMAND ----------

register=mlflow.register_model(
  f"runs:/"+run.info.run_id+"/linker"
  , name='splink_linker'
)

# COMMAND ----------

model_name = register.name
model_version = register.version
loaded_model = mlflow.pyfunc.load_model(
  model_uri=f"models:/{model_name}/{model_version}"
)

# COMMAND ----------

def get_model_json(model_uri): 
  loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)
  temp_file_path = f"/tmp/{model_uri.split('/')[1]}/splink_mlflow_artifacts_download"
  os.makedirs(temp_file_path)
  mlflow.artifacts.download_artifacts(
    artifact_uri=f"models:/{model_name}/{model_version}"
    ,dst_path=temp_file_path
  )
  
  with open(f"{temp_file_path}/artifacts/linker.json", "r") as f:
    linker_json = json.load(f)
  shutil.rmtree(temp_file_path)
  
  return linker_json
  

# COMMAND ----------

get_model_json(f"models:/{model_name}/{model_version}")

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /tmp/splink_linker/

# COMMAND ----------

mlflow.artifacts.download_artifacts(
  artifact_uri=f"models:/{model_name}/{model_version}"
#  ,artifact_path= "linker.json"
  ,dst_path="/tmp"
)

# COMMAND ----------

linker_json

# COMMAND ----------

with open("/tmp/artifacts/linker.json", "r") as f:
  linker_json = json.load(f)
