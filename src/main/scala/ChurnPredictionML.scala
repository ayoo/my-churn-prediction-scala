import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}

object ChurnPredictionML {

  val churnSchema = StructType(Seq(
    StructField("state", StringType, true),
    StructField("account_length", DoubleType, true),
    StructField("area_code",DoubleType, true),
    StructField("phone", StringType, true),
    StructField("int_plan", BooleanType, true),
    StructField("vmail_plan", BooleanType, true),
    StructField("vmail_message", DoubleType, true),
    StructField("day_mins", DoubleType, true),
    StructField("day_calls", DoubleType, true),
    StructField("day_charge", DoubleType, true),
    StructField("eve_mins", DoubleType, true),
    StructField("eve_calls", DoubleType, true),
    StructField("eve_charge", DoubleType, true),
    StructField("night_mins", DoubleType, true),
    StructField("night_calls", DoubleType, true),
    StructField("night_charge", DoubleType, true),
    StructField("int_mins", DoubleType, true),
    StructField("int_calls", DoubleType, true),
    StructField("int_charge", DoubleType, true),
    StructField("custserve_calls", DoubleType, true),
    StructField("label", StringType, true)
  ))

  def main(args: Array[String]) = {

    val conf = new SparkConf().setMaster("local[*]").setAppName("Churn Prediction using ML package")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // x => catching(classOf[Exception]).opt(x.toDouble))
    val rawData = sc.textFile("src/main/resources/churn_no_header.csv")
      .map(line => line.split(","))
      .map { row =>
        row.map { x =>
          try {
            if (x == "yes") true
            else if (x == "no") false
            else x.toDouble
          } catch {
            case _: Throwable => x // just return itself on other types
          }
        }
      }
    val churnRawDF = sqlContext.createDataFrame(rawData.map(Row.fromSeq(_)), churnSchema)
    //println(churnRawDF.show)

    val va = new VectorAssembler().setOutputCol("features")
    va.setInputCols(churnRawDF.columns.diff(Array("state","area_code","phone","label")))
    val churnDF = va.transform(churnRawDF).select("features", "label")
    //println(churnDF.show)

    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(10)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(Array("False.", "True."))

    val Array(trainingData, testData) = churnDF.randomSplit(Array(0.8, 0.2))
    trainingData.cache()
    testData.cache()

    val dt = new RandomForestClassifier() // or DecisionTreeClassifier
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setImpurity("gini")
      .setMaxBins(30)
      .setMaxDepth(10)
      .setNumTrees(15)

    val estimator = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = estimator.fit(trainingData)
    val predictions = model.transform(testData)
    println(predictions.show(50))

    //ROC
    val rocEval = new BinaryClassificationEvaluator().setLabelCol("indexedLabel")
    val rocScore = rocEval.evaluate(predictions)
    println(s"Metric Name: area under ROC, score: $rocScore")

    //PR
    val prEval = new BinaryClassificationEvaluator().setLabelCol("indexedLabel").setMetricName("areaUnderPR")
    val prScore = prEval.evaluate(predictions)
    println(s"Metric Name: area under PR, score: $prScore")

    // Cross Validation

    val cv = new CrossValidator().setEstimator(estimator).setEvaluator(rocEval).setNumFolds(3)
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5,10,15,20))
      .addGrid(dt.numTrees, Array(5,10,15,20))
      .addGrid(dt.maxBins, Array(10,20,30,40))
      .build()
    cv.setEstimatorParamMaps(paramGrid)

    val cvModel = cv.fit(trainingData)

    val cvEval = new BinaryClassificationEvaluator().setLabelCol("indexedLabel")
    val cvScore = cvEval.evaluate(cvModel.bestModel.transform(testData))
    val theClassifier = cvModel.bestModel.parent.asInstanceOf[Pipeline].getStages(2).asInstanceOf[RandomForestClassifier]
    val bestMaxDepth = theClassifier.getMaxDepth
    val bestNumTrees = theClassifier.getNumTrees
    val bestMaxBins = theClassifier.getMaxBins

    println(s"Metric Name: Cross Validated ROC, score: $cvScore")
    println(s"Max Dept: $bestMaxDepth, MaxBins: $bestMaxBins, NumTrees: $bestNumTrees")

    /*
      Metric Name: area under ROC, score: 0.9069377030035408
      Metric Name: area under PR, score: 0.9069377030035413
    -----------------------------------------------------------
      Metric Name: Cross Validated ROC, score: 0.926115835188497
      Max Dept: 10, MaxBins: 40, NumTrees: 20
     */

  }
}
