import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("spark://34.16.229.113:7077").getOrCreate()
rdd=spark.sparkContext.parallelize([1,2,3,4,5])

rddCollect = rdd.collect()
print("Number of Partitions: "+str(rdd.getNumPartitions()))
print("Action: First element: "+str(rdd.first()))
print(rddCollect)

emptyRDD = spark.sparkContext.emptyRDD()
emptyRDD2 = rdd=spark.sparkContext.parallelize([])

print(str(emptyRDD2.isEmpty()))
