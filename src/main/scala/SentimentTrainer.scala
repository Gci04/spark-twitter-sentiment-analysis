import java.util.Locale

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.IntegerType

object SentimentTrainer {
  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Spark Sentiment")
      .config("spark.master", "local")
      .getOrCreate()

    val twitterPath = args(0)

    val twitterTrainPath = args(0) + "/train.csv"
//    val twitterTestPath = args(0) + "/test.csv"

    val twitterData = readTwitterData(twitterTrainPath, spark)
//    val testdata = readTwitterData(twitterTestPath, spark)

    val tokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized All")
      .setPattern("\\s+")

    val wordTokenizer = new RegexTokenizer()
      .setInputCol("Preprocessed")
      .setOutputCol("Tokenized Words")
      .setPattern("\\W")

    Locale.setDefault(Locale.ENGLISH)

    val stopW = new StopWordsRemover()
      .setInputCol("Tokenized Words")
      .setOutputCol("Stopped")

    val ngram = new NGram()
      .setN(2)
      .setInputCol("Stopped")
      .setOutputCol("Grams")

    val tokenVectorizer = new CountVectorizer()
      .setInputCol("Tokenized All")
      .setOutputCol("Token Vector")

    val gramVectorizer = new CountVectorizer()
      .setInputCol("Grams")
      .setOutputCol("Gram Vector")

    val assembler = new VectorAssembler()
      .setInputCols(Array("Token Vector"))
      .setOutputCol("features")

    val model = new LogisticRegression()
      .setFamily("multinomial")
      .setLabelCol("Sentiment")

    val pipe = new Pipeline()
      .setStages(Array(tokenizer,
        wordTokenizer, stopW,
        ngram, tokenVectorizer,
        gramVectorizer,
        assembler, model))

    val paramMap = new ParamMap()
      .put(tokenVectorizer.vocabSize, 10000)
      .put(gramVectorizer.vocabSize, 10000)
      .put(model.elasticNetParam, .8)
      .put(model.tol, 1e-20)
      .put(model.maxIter, 100)
//
    val lr = pipe.fit(twitterData, paramMap)
//
    val tr = lr.transform(twitterData).select("Sentiment", "probability", "prediction")
    tr.take(10).foreach(println)

    val eval = new BinaryClassificationEvaluator()
      .setLabelCol("Sentiment")
      .setRawPredictionCol("prediction")

    val roc = eval.evaluate(tr)
    println(s"ROC: ${roc}")

    tr.printSchema()
    tr.printSchema()

    val paramGrid = new ParamGridBuilder()
      .addGrid(tokenVectorizer.vocabSize, Array(10000))
      .addGrid(gramVectorizer.vocabSize, Array(10000))
      .addGrid(model.elasticNetParam, Array(.8))
      .addGrid(model.tol, Array(1e-20))
      .addGrid(model.maxIter, Array(100))
      .build()

    val cv = new CrossValidator()
          .setEstimator(pipe)
          .setEvaluator(new BinaryClassificationEvaluator()
            .setRawPredictionCol("prediction")
            .setLabelCol("Sentiment"))
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(5)
          .setParallelism(1)

    val cvmodel = cv.fit(twitterData)

    cvmodel.transform(twitterData)
          .select("ItemID","Preprocessed", "probability", "prediction")
          .collect().take(10)
          .foreach(println)

    println("Metrics: \n\n\n")
    cvmodel.avgMetrics.foreach(println)
    println("\n\n\n")

    cvmodel.write.overwrite().save("sentiment-classifier")
  }

  def readTwitterData(path: String, spark: SparkSession) = {

    val data = spark.read.format("csv")
      .option("header", "true")
      .load(path)

    val preprocess: String => String = {
      _.replaceAll("((.))\\1+","$1")
    }
    val preprocessUDF = udf(preprocess)

    val newCol = preprocessUDF.apply(data("SentimentText"))
    val label = data("Sentiment").cast(IntegerType)

    data.withColumn("Preprocessed", newCol)
      .withColumn("Sentiment",label)
      .select("ItemID","Sentiment","Preprocessed")

  }

}
