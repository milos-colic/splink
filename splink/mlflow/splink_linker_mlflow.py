import mlflow
import mlflow.pyfunc
import os
import random
import splink.spark.spark_comparison_library as cl
import tempfile

from .util import *
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score, f1_score
from splink.spark.spark_linker import SparkLinker


class SplinkLinkerModel(mlflow.pyfunc.PythonModel):
    """
    MLFlow compatibility wrapper for splink linker.
    It enables mlflow.pyfunc.log_model(...) and mlflow.pyfunc.load_model(...) APIs.
    The model instance handles the fact spark inst pickle serialisable and nullifies spark when serialisation occurs.
    """
 
    def __init__(self, **kwargs):
        """
        Default constructor for model instace.
        It allows additional parameters to be provided via kwargs.
        """
        self.settings = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __getstate__(self):
        """
        Serialisation logic for pickle processes.
        Makes sure linker object is nullified to avoid spark serialisation issues.
        Serialise model as its json settings object.
        """
        json_result = self.get_settings_object()
        self.linker = None
        return json_result
            
    def __setstate__(self, json_dict):
        """
        Deserialise logic for pickle processes.
        Set settings of the model.
        """
        self.settings = json_dict
    
    def load_context(self, context):
        self.context = context
        
    def clear_context(self):
        self.linker.spark = None
        
    def set_should_evaluate(self, flag):
        """
        Mechanism to avoid evaluate step in the training if no labelled data is provided.
        If the flag is True evaluate method is a noop.
        """
        self.should_evaluate = flag
        
    def spark_linker(self, data):
        """
        Initialises SparkLinker instance using provided data and current active spark session.
        Parameters
        ----------
        data:
            Data which we are deduping using Splink.
        Returns
            Initialised SplinkLinker instance.
        -------
        """
        spark = SparkSession.builder.getOrCreate()
        self.linker = SparkLinker(data, spark=spark)
        spark = None
        return self.linker
        
    def set_spark_linker(self, linker: SparkLinker):
        """
        Set linker to provided instance.
        """
        self.linker = linker
    
    def set_settings(self, settings):
        """
        Set settings to provided instance.
        Initialised the available linker instance.
        """
        self.settings = settings
        if self.linker:
            self.linker.initialise_settings(self.settings)
        
    def set_blocking_rules(self, blocking_rules):
        """
        Update settings with provided blocking rules.
        Initialise the available linker instance with updated settings.
        """
        self.settings.update({"blocking_rules_to_generate_predictions": blocking_rules})
        if self.linker:
            self.linker.initialise_settings(self.settings)
    
    def set_comparisons(self, comparisons):
        """
        Update settings with provided comparisons.
        Initialise the available linker instance with updated settings.
        """
        self.settings.update({"comparisons": comparisons})
        if self.linker:
            self.linker.initialise_settings(self.settings)
            
    def set_stage1_columns(self, columns):
        """
        Set the columns to be used in the first stage of the m estimation.
        """
        self.stage1_columns = columns
        
    def set_stage2_columns(self, columns):
        """
        Set the columns to be used in the second stage of the m estimation.
        """
        self.stage2_columns = columns
    
    def set_target_rows(self, rows):
        """
        Set the number of rows to be used for random matching probability estimation.
        """
        self.target_rows = rows
        
    def get_linker(self):
        """
        Get the linker instance.
        """
        return self.linker
    
    def get_settings(self):
        """
        Get the settings instance.
        """
        return self.settings
   
    def get_settings_object(self):
        """
        Get the initialised settings instance which will contain trained params.
        Parameters
        ----------
        Returns
            The json settings of the linker as a dict.
        -------
        """
        return self.get_linker()._settings_obj.as_dict()
    
    def estimate_random_match_probability(self, rules, recall):
        """
        Runs estimation of the probability of a random match of two records.
        Parameters
        ----------
        rules:
            Deterministic blocking rules.
        recall:
            Recall for the estimation.
        Returns
        -------
        """
        self.linker.estimate_probability_two_random_records_match(rules, recall=recall)
        
    def estimate_u(self, target_rows):
        """
        Runs estimation of the u probability.
        Parameters
        ----------
        target_rows:
            Number of rows to be used for u estimation.
        Returns
        -------
        """
        self.linker.estimate_u_using_random_sampling(target_rows=target_rows)
        
    def estimate_m(self, data, columns_1, columns_2):
        """
        Runs estimation of the m probability via two stage approach.
        Parameters
        ----------
        columns_1:
            Columns to be used in the first stage of m estimation.
        columns_2:
            Columns to be used in the second stage of m estimation.
        Returns
        -------
        """
        fixed_columns = random.choices(data.columns, k=2)
        training_rule_1 = " and ".join([f"(l.{cn} = r.{cn})" for cn in columns_1])
        training_rule_2 = " and ".join([f"(l.{cn} = r.{cn})" for cn in columns_2])
        self.linker.estimate_parameters_using_expectation_maximisation(training_rule_1)
        self.linker.estimate_parameters_using_expectation_maximisation(training_rule_2)
        
    def fit(self, X):
        """
        Model training given input parameters and a training dataset.
        """
        if self.settings is None:
            raise Exception("Cannot initialise linker without settings being set. Please use model.set_settings.")
        deterministic_rules = [
          "l.name = r.name and levenshtein(r.date_of_birth, l.date_of_birth) <= 1",
          "l.address_line_1 = r.address_line_1 and levenshtein(l.name, r.name) <= 5",
          "l.name = r.name and levenshtein(l.address_line_1, r.address_line_1) <= 5",
        ]
      
        linker = self.spark_linker(X)
        linker.initialise_settings(self.settings)
        self.set_spark_linker(linker)
        self.estimate_random_match_probability(deterministic_rules, 0.8)
        self.estimate_u(self.target_rows)
        self.estimate_m(X, self.stage1_columns, self.stage2_columns)
        return linker
    
    def log_settings_as_json(self, path):
        """
        Simple method for logging a splink model
        Parameters
        ----------
        linker : Splink model object
        Returns
        -------
        """
        path = "linker.json"
        if os.path.isfile(path):
            os.remove(path)
        self.linker.save_settings_to_json(path)
        mlflow.log_artifact(path)
        os.remove(path)
        
    def _log_chart(self, chart_name, chart):
        '''
        Save a chart to MLFlow. This writes the chart out temporarily for MLFlow to pick up and save.
        Parameters
        ----------
        chart_name : str, the name the chart will be given in MLFlow
        chart : chart object from Splink
        Returns
        -------
        '''
        path = f"{chart_name}.html"
        if os.path.isfile(path):
            os.remove(path)
        save_offline_chart(chart.spec, path)
        mlflow.log_artifact(path)
        os.remove(path)
  
    def evaluate(self, y_test, y_test_pred):
        """
        Evaluate model performance.
        """
        if self.should_evaluate:
            spark = SparkSession.builder.getOrCreate()
            splink_tables = spark.sql('show tables like "*__splink__*"')
            temp_tables = splink_tables.collect()
            drop_tables = list(map(lambda x: x.tableName, temp_tables))
            for x in drop_tables:
                spark.sql(f"drop table {x}")
            self.linker.register_table(y_test, "labels")
            roc_auc = linker.roc_chart_from_labels_table("labels")
            return roc_auc
        else:
            return None
 
    def predict(self, context, X):
        """
        Predict labels on provided data
        """
        linker = self.spark_linker(X)
        linker.initialise_settings(self.settings)
        result = linker.predict()
        return result
    