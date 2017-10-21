import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.regression.LabeledPoint

//classifier
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

object ChurnPredictionMLLib {

  def main(args: Array[String]){
    val conf = new SparkConf().setMaster("local[*]").setAppName("My First Churn Prediction")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    //import sqlContext.implicits._

    val exIndex = Seq(0,2,3)
    val data = sc.textFile("src/main/resources/churn_no_header.csv")
        .map(line => line.split(","))
        .map(splitted => splitted.filter(e => !exIndex.contains(splitted.indexOf(e))))
        .map(dropped => dropped.map { d =>
          if (d == "yes" || d == "True.")
            1
          else if (d == "no" || d == "False.")
            0
          else
            d.toDouble
        })
        .map { converted => (converted.last, Vectors.dense(converted.dropRight(1)))}

    val sets = data.randomSplit(Array(0.8, 0.2))
    val trainingSet = sets(0)
    val testSet = sets(1)
    val trainingVectors = trainingSet.map(_._2)
    val scaler = new StandardScaler(withMean=true, withStd=true).fit(trainingVectors)
    val scaledTrainingData = trainingSet.map(d => LabeledPoint(d._1, scaler.transform(d._2)))
    val scaledTestData = testSet.map(d => LabeledPoint(d._1, scaler.transform(d._2)))
    val trainingData = trainingSet.map(d => LabeledPoint(d._1, d._2))
    val testData = testSet.map(d => LabeledPoint(d._1, d._2))

    val total = scaledTestData.count()
    val numIterations = 10
    val maxTreeDepth = 5

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3
    val featureSubsetStrategy = "auto"
    val impurity = "gini" // "entropy"
    val maxDepth = 5
    val maxBins = 32

    val lrModel = LogisticRegressionWithSGD.train(scaledTrainingData, numIterations)
    val svmModel = SVMWithSGD.train(scaledTrainingData, numIterations)
    val dtModel = DecisionTree.train(trainingData, Algo.Classification, Entropy, maxTreeDepth)
    val rfModel = RandomForest.trainClassifier(
      input = trainingData,
      numClasses = numClasses,
      categoricalFeaturesInfo = categoricalFeaturesInfo,
      numTrees = numClasses,
      featureSubsetStrategy = featureSubsetStrategy,
      impurity = impurity,
      maxDepth = maxDepth,
      maxBins = maxBins)


    //Prediction and Evaluation

    // Logistic Regression
    val lrPredLabels = scaledTestData.map { point =>
      (lrModel.predict(point.features), point.label)
    }
    val lrMetrics = new BinaryClassificationMetrics(lrPredLabels)
    val lrPR = lrMetrics.areaUnderPR()
    val lrROC = lrMetrics.areaUnderROC()
    println(s"The Logistic Regression: areaUnderPR=$lrPR, areaUnderROC=$lrROC")

    // Support Vector Machine
    val svmPredLabels = scaledTestData.map { point =>
      (svmModel.predict(point.features), point.label)
    }
    val svmMetrics = new BinaryClassificationMetrics(svmPredLabels)
    val svmPR = svmMetrics.areaUnderPR()
    val svmROC = svmMetrics.areaUnderROC()
    println(s"Support Vector Machine: areaUnderPR=$svmPR, areaUnderROC=$svmROC")

    // Decision Tree
    val dtPredLabels = testData.map { point =>
      (dtModel.predict(point.features), point.label)
    }
    val dtMetrics = new BinaryClassificationMetrics(dtPredLabels)
    val dtPR = dtMetrics.areaUnderPR()
    val dtROC = dtMetrics.areaUnderROC()
    println(s"Decision Tree: areaUnderPR $dtPR, areaUnderROC $dtROC")

    // Random Forest
    val rfPredLabels = testData.map { point =>
      (rfModel.predict(point.features), point.label)
    }
    val rfMetrics = new BinaryClassificationMetrics(rfPredLabels)
    val rfPR = rfMetrics.areaUnderPR()
    val rfROC = rfMetrics.areaUnderROC()
    println(s"Random Forest: areaUnderPR $rfPR, areaUnderROC $rfROC")
    sc.stop()
  }
}
