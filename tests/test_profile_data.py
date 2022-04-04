import sqlite3
import pandas as pd

from splink.duckdb.duckdb_linker import DuckDBLinker
from splink.sqlite.sqlite_linker import SQLiteLinker
from splink.spark.spark_linker import SparkLinker

from basic_settings import get_settings_dict


def test_profile_using_duckdb():
    df = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    settings_dict = get_settings_dict()
    linker = DuckDBLinker(
        settings_dict, input_tables={"fake_data_1": df}, connection=":memory:"
    )

    linker.profile_columns(
        ["first_name", "surname", "first_name || surname", "concat(city, first_name)"],
        top_n=15,
        bottom_n=5,
    )


def test_profile_using_sqlite():
    df = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")

    con = sqlite3.connect(":memory:")

    df.to_sql("fake_data_1", con, if_exists="replace")
    settings_dict = get_settings_dict()
    linker = SQLiteLinker(
        settings_dict,
        connection=con,
        input_tables={"fake_data_1": "fake_data_1"},
    )

    linker.profile_columns(["first_name", "surname", "first_name || surname"])


# @pytest.mark.skip(reason="Uses Spark so slow and heavyweight")
def test_profile_using_spark(df_spark):
    settings_dict = get_settings_dict()
    linker = SparkLinker(settings_dict, input_tables={"fake_data_1": df_spark})

    linker.profile_columns(
        ["first_name", "surname", "first_name || surname", "concat(city, first_name)"]
    )
