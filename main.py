from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import LongType
from sparkml_utils import *

import findspark

import pandas as pd

import seaborn as sns

findspark.init("C:\Program Files\Spark\spark-3.3.1-bin-hadoop3")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option('display.width', 1000)

spark = SparkSession.builder \
    .appName("KMeans RFM Analyze with SparkML Pipeline") \
    .master("local[2]") \
    .config("spark.sql.adaptive.enabled", True) \
    .getOrCreate()
    
# hdfs dfs -put /users/talha/datasets/flo100k.csv /user/talha/datasets

df = spark.read \
    .format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("sep", "|") \
    .load("file:///Users/talha/OneDrive/Masaüstü/Talha Nebi Kumru/Data Enginnering/Miuul/SparkML/KMeans_RFM_Analyze_with_SparkML/KMeans_RFM_Analyze_with_SparkML/datasets/flo100k.csv")
    
df.limit(5).toPandas()

#########################################
# Dataframe Dtypes
#########################################

df.printSchema()
df.dtypes

"""
root
 |-- master_id: string (nullable = true)
 |-- order_channel: string (nullable = true)
 |-- platform_type: string (nullable = true)
 |-- last_order_channel: string (nullable = true)
 |-- first_order_date: timestamp (nullable = true)
 |-- last_order_date: timestamp (nullable = true)
 |-- last_order_date_online: timestamp (nullable = true)
 |-- last_order_date_offline: timestamp (nullable = true)
 |-- order_num_total_ever_online: integer (nullable = true)
 |-- order_num_total_ever_offline: integer (nullable = true)
 |-- customer_value_total_ever_offline: double (nullable = true)
 |-- customer_value_total_ever_online: double (nullable = true)
 |-- interested_in_categories_12: string (nullable = true)
 |-- online_product_group_amount_top_name_12: string (nullable = true)
 |-- offline_product_group_name_12: string (nullable = true)
 |-- last_order_date_new: timestamp (nullable = true)
 |-- store_type: string (nullable = true)
"""

#########################################
# Trim Strings
#########################################

df = trim_string(df)

#########################################
# Null Check
#########################################

print_null(df)

"""
last_order_date_online has 70784 -  70.78 % null count.
last_order_date_offline has 21703 -  21.70 % null count.
order_num_total_ever_online has 70784 -  70.78 % null count.
order_num_total_ever_offline has 21703 -  21.70 % null count.
interested_in_categories_12 has 56590 -  56.59 % null count.
online_product_group_amount_top_name_12 has 88295 -  88.30 % null count.
offline_product_group_name_12 has 77209 -  77.21 % null count.
"""

df.selectExpr("AVG(order_num_total_ever_online)").show() # 2.67
df.selectExpr("AVG(order_num_total_ever_offline)").show() # 1.62

df = update_null(df, "order_num_total_ever_online", "int", 0)
df = update_null(df, "order_num_total_ever_offline", "int", 0)

#########################################
# Unique Control
#########################################

df.select("master_id").toPandas()["master_id"].is_unique

#########################################
# Data Understanding
#########################################

df.groupBy(F.col("order_channel")) \
    .agg(F.count("platform_type").alias("Count")).show()
    
#########################################
# Data Analyze
#########################################

df = df.withColumn("order_num_total", 
                   F.col("order_num_total_ever_online") + 
                   F.col("order_num_total_ever_offline"))

df = df.withColumn("customer_value_total",
                   F.col("customer_value_total_ever_offline") +
                   F.col("customer_value_total_ever_online"))

df.sort("customer_value_total", ascending=True).groupBy("order_channel") \
    .agg(F.count("platform_type").alias("Customer Count"),
         F.mean("order_num_total").alias("Mean"),
         F.percentile_approx("customer_value_total", 0).alias("%0"),
         F.percentile_approx("customer_value_total", 0.25).alias("%25"),
         F.percentile_approx("customer_value_total", 0.50).alias("%50"),
         F.percentile_approx("customer_value_total", 0.75).alias("%75"),
         F.percentile_approx("customer_value_total", 1).alias("%100")).show()
    
"""
+-------------+--------------+------------------+-----+------+------+------+--------+
|order_channel|Customer Count|              Mean|   %0|   %25|   %50|   %75|    %100|
+-------------+--------------+------------------+-----+------+------+------+--------+
|  Android App|         11989|  3.50971724080407|17.14|175.99|334.94|656.42|21962.85|
|       Mobile|          8512| 2.798637218045113| 9.99|139.98|259.94|469.86|17622.99|
|      Ios App|          3964| 3.377648839556004|20.99|188.99|353.97| 687.9| 8285.71|
|      Desktop|          4751| 2.538623447695222| 9.96|149.99|257.92|443.97| 7148.97|
|      Offline|         70784|1.6003475361663653|  0.0| 89.99|149.99|269.97|51178.23|
+-------------+--------------+------------------+-----+------+------+------+--------+
"""

df.sort("customer_value_total", ascending=True).groupBy("platform_type") \
    .agg(F.count("platform_type").alias("Customer Count"),
         F.mean("order_num_total").alias("Mean"),
         F.percentile_approx("customer_value_total", 0).alias("%0"),
         F.percentile_approx("customer_value_total", 0.25).alias("%25"),
         F.percentile_approx("customer_value_total", 0.50).alias("%50"),
         F.percentile_approx("customer_value_total", 0.75).alias("%75"),
         F.percentile_approx("customer_value_total", 1).alias("%100")).show()
    
"""
+-------------+--------------+------------------+----+------+------+------+--------+
|platform_type|Customer Count|              Mean|  %0|   %25|   %50|   %75|    %100|
+-------------+--------------+------------------+----+------+------+------+--------+
|  OmniChannel|         12569|3.4947092051873656| 0.0|174.98|334.98|613.44|51178.23|
|       Online|         21440|2.6207089552238805|9.96|131.99|243.48|459.98|21962.85|
|      Offline|         65991|1.5837917291751906| 0.0| 89.99|149.99| 269.0|37141.97|
+-------------+--------------+------------------+----+------+------+------+--------+
"""

#########################################
# RFM Analyze
#########################################

rfm = df.select("master_id", "last_order_date_new", "order_num_total", "customer_value_total")

rfm.limit(5).toPandas()
rfm.show(5)
rfm.printSchema()

rfm.selectExpr("MAX(last_order_date_new)").show() # 2021-05-30
rfm = rfm.withColumn("today_date", F.lit("2021-06-01"))
rfm = rfm.withColumn("today_date", F.to_timestamp("today_date", "yyyy-MM-dd"))

rfm = rfm.withColumn("recency", 
               F.col("today_date").cast(LongType()) - F.col("last_order_date_new").cast(LongType())) \
    .withColumn("recency", F.round(F.col("recency") / (24 * 3600)))
    
rfm = rfm.drop("today_date")
rfm = rfm.drop("last_order_date_new")

rfm = rfm.withColumnRenamed("order_num_total", "frequency")
rfm = rfm.withColumnRenamed("customer_value_total", "monetary")

#########################################
# Model Preparation
#########################################

numeric_cols = ["recency", "frequency", "monetary"]

assembler = set_assembler(numeric_cols)

scaler = set_scaler()

pipeline_object = set_pipeline(False, [assembler, scaler])

pipeline_model = pipeline_object.fit(rfm)

pipeline_df = pipeline_model.transform(rfm)
pipeline_df.show(5)

kmeans_elbow(pipeline_df, range_=(2, 25))

kmeans_model = compute_kmeans(pipeline_df, 5)

transformed_df = kmeans_model.transform(pipeline_df)
transformed_df.show(5)



pandas_df = transformed_df.toPandas()

pandas_df.groupby("prediction").describe().T

sns.histplot(data=pandas_df, x="prediction", bins=30, hue="prediction")

sns.boxplot(x="prediction", y="recency",
            data=pandas_df)

sns.boxplot(x="prediction", y="frequency",
            data=pandas_df)

sns.boxplot(x="prediction", y="monetary",
            data=pandas_df)

sns.scatterplot(x="recency", y="monetary",
                data=pandas_df,
                hue="prediction",
                palette="deep")

sns.scatterplot(x="recency", y="frequency",
                data=pandas_df,
                hue="prediction",
                palette="deep")

sns.scatterplot(x="monetary", y="frequency",
                data=pandas_df,
                hue="prediction",
                palette="deep")

spark.stop()
