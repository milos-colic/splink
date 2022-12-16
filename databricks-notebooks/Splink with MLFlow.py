# Databricks notebook source
pip install git+https://github.com/robertwhiffin/splink.git@splink-databricks

# COMMAND ----------

import requests
import time
from pyspark.sql import functions as F
from pyspark.sql import Window
import splink.spark.spark_comparison_library as cl
import pandas as pd 
from pyspark.sql import types
from pyspark.sql.functions import collect_set, size
import mlflow

# COMMAND ----------

experiment
