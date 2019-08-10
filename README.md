# Spark twitter sentiment analysis
This Repository implements a machine learning model which analyzes tweets and predicts if they are positive, or negative. The programming laguage is scala. 

## Dataset Description
In this repository Twitter [dataset from Kaggle](https://www.kaggle.com/c/twitter-sentiment-analysis2/data) is used. The training set contains 100k examples, test set has 300k examples. The data is provided in CSV format. Data is very irregular and requires preprocessing
```
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - +
| ItemID | Sentiment |       SentimentText              |
+ - - - -+ - - - - - + - - - - - - - - - - - - - - - - -+
|   1    |     0     | is so sad for my APL friend..... |
|   2    |     0     | I missed the New Moon trailer... |
|   3    |     1     |    omg its already 7:30 :O       |
|  ...   |    ...    |    .......................       |
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - +
```
## Training process structure

The flow of the whole model development is outlined by the shceme below 

<p>
<img src="https://github.com/Gci04/spark-twitter-sentiment-analysis/blob/master/NLgewhb.png" alt="Scheme" width="550"/>
</p>

The schema is implemented as pipeline in file.scala

## ML Model

Machine Learning model used as a classifier is Logistic Regression. Using spark  parameter grid map and CrossValidator() model is tuned and crossvalidated in parallel

### Distributed Hyperparameter Search

[parameter grid map](https://spark.apache.org/docs/2.2.0/ml-tuning.html#model-selection-aka-hyperparameter-tuning)

```scala
val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(ngramVectorizer.vocabSize, 10000)
      .put(classifier.tol, 1e-20)
      .put(classifier.maxIter, 100)
      
val model = pipe.fit(twitterData, paramMap)
```

```scala
val paramGrid = new ParamGridBuilder()
      .addGrid(tokenVectorizer.vocabSize, Array(10000, 20000))
      .addGrid(gramVectorizer.vocabSize, Array(10000, 15000))
      .addGrid(lr.tol, Array(1e-20, 1e-10, 1e-5))
      .addGrid(lr.maxIter, Array(100, 200, 300))
      .build()
```

### Cross Validation

[k-fold cross validation prosedure](https://spark.apache.org/docs/2.2.0/ml-tuning.html#cross-validation).

```scala
val cv = new CrossValidator()
      .setEstimator(pipe)
      .setEvaluator(new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol("Sentiment"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(2)
```
## Performance Measurement 

[Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is used to measure model performance.

```scala
val eval = new BinaryClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setRawPredictionCol("prediction")

val roc = eval.evaluate(tr)
println(s"ROC: ${roc}")
```

## Configuring and Run on Cluster
Using intellij idea having build.sbt file a .jar file can be easily compiled and deployed in cluster using the following comand: 

```bash
spark-submit --master yarn --deploy-mode client path/to/jar hdfs://twitter/
```

