from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt

def null_count(df, col):
    return df.select(col).filter((F.col(col) == "NA") |
                                (F.col(col) == "") |
                                (F.col(col).isNull())).count()

def print_null(df):
    for col in df.columns:
        nc = null_count(df, col)
        count = df.count()
        print(f"{col} has {nc} - {(nc / count) * 100 : .2f} % null count.") if nc > 0 else None
        
def trim_string(df):
    for col in df.dtypes:
        df = df.withColumn(col[0], F.trim(col[0])) if col[1] == 'string' else df
        
    return df

def update_null(df, col, type_, avg=0):
    return df.withColumn(col, F.when((F.col(col) == "NA") |
                                (F.col(col) == "") |
                                (F.col(col).isNull()), avg) \
        .otherwise(F.col(col))) \
        .withColumn(col, F.col(col).cast(type_))
        
def set_assembler(*args):
    empty = []
    for l in args:
        empty += l
        
    return VectorAssembler() \
        .setHandleInvalid("skip") \
        .setInputCols(empty) \
        .setOutputCol("unscaled_features")
        
def set_scaler():
    return StandardScaler() \
        .setInputCol("unscaled_features") \
        .setOutputCol("features")
        
def set_pipeline(string_indexer=True, *args):
    if string_indexer:
        string_indexer_object = args[0]
        
        args = args[1:] if len(args) > 1 else args
    
        pipe_list = []
        for p in args:
            pipe_list += p
            
        return Pipeline() \
            .setStages(string_indexer_object + pipe_list)
    else:
        pipe_list = []
        for p in args:
            pipe_list += p
            
        return Pipeline() \
            .setStages(pipe_list)
            
def compute_kmeans(df, k):
    kmeans_obj = KMeans() \
        .setSeed(142) \
        .setK(k)
        
    return kmeans_obj.fit(df)

def kmeans_elbow(df, range_=(2, 11)):
    costs = {}
    evaluator = ClusteringEvaluator()
    for k in range(range_[0], range_[1]):
        kmeans_model = compute_kmeans(df, k)
        
        transformed_df = kmeans_model.transform(df)
        
        score = evaluator.evaluate(transformed_df)
        
        costs[k] = score
        
        print(f"k : {k}, score: {score}")
    
    plt.plot(costs.keys(), costs.values(), "bx-")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title("Kmeans Elbow Graph")
