from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as f

from splink import Splink
from .blocking import (
    sql_gen_comparison_columns, 
    _get_columns_to_retain_blocking
)
from .model import Model
from .validate import validate_input_datasets
from .gammas import _sql_gen_add_gammas
from .iterate import iterate
from .charts import altair_if_installed_else_json

def self_link(linker: Splink):

    # Take df and settings from Splink object
    df = linker.df
    settings = linker.settings_dict
    spark = linker.spark

    unique_id_col = settings["unique_id_column_name"]

    # Change settings to link each record in df to itself
    settings["link_type"] = "link_only"
    settings["max_iterations"] = 0
    settings["retain_matching_columns"] = True
    settings["blocking_rules"] = [f'l.{unique_id_col} = r.{unique_id_col}']

    model = Model(settings, spark)

    df.createOrReplaceTempView("df")

    columns_to_retain = _get_columns_to_retain_blocking(settings, df)

    # Fudge df_comparisons (because block_using_rules doesn't quite work)
    sql = f"""select
        {sql_gen_comparison_columns(columns_to_retain)},
        '0' as match_key
        from df as l
        inner join df as r
        on {settings['blocking_rules'][0]}
    """

    df_comparison = spark.sql(sql)
    df_comparison.createOrReplaceTempView("df_comparison")

    sql = _sql_gen_add_gammas(settings, df_comparison)
    df_gammas = spark.sql(sql)

    # df_e with same no. of rows as df, with a self-link score for each record
    df_e = iterate(df_gammas, model, spark)

    return df_e

def unlinkables_chart(df_self: DataFrame, spark: SparkSession, x_col="match_weight", source_dataset=None):

    if x_col not in ["match_weight", "match_probability"]:
        raise ValueError(
            f"{x_col} must be 'match_weight' (default) or 'match_probability'."
        )

    chart_path = "unlinkables_chart_def.json"
    unlinkables_chart_def = load_chart_definition(chart_path)


    data = df_self.groupBy(f.round("match_probability",5).alias("match_probability"))\
        .agg(
            f.count("match_weight").alias("count"), 
            f.max(f.round("match_weight",2)).alias("match_weight")
        ).toPandas()
    data = data.sort_values("match_probability").reset_index(drop=True)
    data["prop"]= data["count"]/ data["count"].sum()
    data["cum_prop"] = data["prop"].cumsum()
    data = data[:-1]

    unlinkables_chart_def["data"]["values"] = data.to_dict('records')

    if x_col == "match_probability":
        unlinkables_chart_def["layer"][0]["encoding"]["x"]["field"] = 'match_probability'
        unlinkables_chart_def["layer"][0]["encoding"]["x"]["axis"]["title"] = 'Threshold match probability'
        unlinkables_chart_def["layer"][0]["encoding"]["x"]["axis"]["format"] = '.2'

        unlinkables_chart_def["layer"][1]["encoding"]["x"]["field"] = 'match_probability'
        unlinkables_chart_def["layer"][1]["selection"]["selector112"]["fields"] = ['match_probability', 'cum_prop']

        unlinkables_chart_def["layer"][2]["encoding"]["x"]["field"] = 'match_probability'
        unlinkables_chart_def["layer"][2]["encoding"]["x"]["axis"]["title"] = 'Threshold match probability'

        unlinkables_chart_def["layer"][3]["encoding"]["x"]["field"] = 'match_probability'

    if source_dataset:
        unlinkables_chart_def["title"]["text"] += f" - {source_dataset}"

    return altair_if_installed_else_json(unlinkables_chart_def)