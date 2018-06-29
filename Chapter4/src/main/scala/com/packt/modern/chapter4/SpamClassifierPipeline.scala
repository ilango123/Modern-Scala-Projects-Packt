package com.packt.modern.chapter4

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.{DataFrame, DataFrameNaFunctions, Row, SparkSession}
import org.apache.spark.ml.feature.{HashingTF, IDF, Normalizer, Tokenizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.explode

/*
  How to run it:
  sbt console
  scala> import com.packt.modern.chapter4.SpamClassifierPipeline
  scala> SpamClassifierPipeline.main(Array("inbox2.txt"))
  //Next time, try loading it on HDFS

*/

case class LabeledHamSpam(label: Double, mailSentence: String)

object SpamClassifierPipeline extends SpamWrapper with SpamLogger {

  def main(args: Array[String]): Unit = {

    //defined class LabeledHamSpam

    val reg1 = raw"[^A-Za-z0-9\s]+" // remove punctuation with numbers
    val reg2 = raw"[^A-Za-z\s]+" // remove punctuation not include numbers
    //val reg = raw"[^A-Za-z\s]+" // remove punctuation not include numbers
    //val lines = sc.textFile("/user/hadoop/netapp_boiler_top20000_np.csv").map(_.replaceAll(reg, "").trim.toLowerCase).toDF("line")

    //val hamRDD = session.sparkContext.textFile("inbox2.txt")
    val hamRDD: org.apache.spark.rdd.RDD[String] = session.sparkContext.textFile(hamSetFileName)
    //hamRDisDset: org.apache.spark.rdd.RDD[String] = inbox.txt MapPartitionsRDD[1] at textFile at <console>:23
    val hamRDD2 = hamRDD.map(_.replaceAll(reg1, "").trim.toLowerCase)
    val hamRDD3: RDD[LabeledHamSpam] = hamRDD2.repartition(4).map(w => LabeledHamSpam(0.0,w))
    //hamRDisDset2: org.apache.spark.rdd.RDD[LabeledHamSpam] = MapPartitionsRDD[14] at map at <console>:28
    hamRDD3.take(10)
    println("The HAM RDD looks like: " + hamRDD3.collect())


    val spamRDD = session.sparkContext.textFile(spamFileName)
    //spamRDisDset: org.apache.spark.rdd.RDD[String] = junk2.txt MapPartitionsRDD[3] at textFile at <console>:23
    val spamRDD2 = spamRDD.map(_.replaceAll(reg1, "").trim.toLowerCase)
    val spamRDD3 = spamRDD2.repartition(4).map(w => LabeledHamSpam(1.0,w))

    //A check
    spamRDD3.take(10)

    val hamAndSpam: org.apache.spark.rdd.RDD[LabeledHamSpam] =  (hamRDD3 ++ spamRDD3)
    //hamAndSpam: org.apache.spark.rdd.RDD[LabeledHamSpam] = UnionRDD[20] at $plus$plus at <console>:34

    hamAndSpam.take(10)
    //res2: Array[LabeledHamSpam] = Array(LabeledHamSpam(0.0,Urgent! Please call 09061743811 from landline. Your ABTA complimentary 4* Tenerife Holiday or ú5000 cash await collection SAE T&Cs Box 326 CW25WX 150ppm), LabeledHamSpam(0.0,PRIVATE! Your 2003 Account Statement for 07815296484 shows 800 un-redeemed S.I.M. points. Call 08718738001 Identifier Code 41782 Expires 18/11/04 ), LabeledHamSpam(0.0,FREE MSG:We billed your mobile number by mistake from shortcode 83332.Please call 08081263000 to have charges refunded.This call will be free from a BT landline), LabeledHamSpam(0.0,complimentary 4 STAR Ibiza Holiday or ú10,000 cash needs your URGENT collection. 09066364349 NOW from Landline not to lose out! Box434SK38WP150PPM18+), LabeledHamSpam(0.0,GENT! We are trying to contact you. Last weeken...

    ////////////////////////////// STEP 1 //////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    //Or use this
    val dataFrame1 = session.createDataFrame(hamAndSpam).toDF("label","lowerCasedSentences")
    //dataFrame1: org.apache.spark.sql.DataFrame = [label: double, features: string]


    val lowerCasedDataFrame = dataFrame1.select(dataFrame1("lowerCasedSentences"), dataFrame1("label"))
    //dataFrame2: org.apache.spark.sql.DataFrame = [features: string, label: double]*/

    //The above did not work as it produces arrays
    //So, this resources seem to work: 1) https://docs.hortonworks.com/HDPDocuments/HDP2/HDP-2.6.4/bk_spark-component-guide/content/spark-dataframe-api.html
    // 2) https://stackoverflow.com/questions/44174747/spark-dataframe-collect-vs-select?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    //The google search term was: "Spark Dataframe select"

    println("lowerCasedDataFrame looks like this:")
    lowerCasedDataFrame.show
    /*
 +--------------------+-----+
| lowerCasedSentences|label|
+--------------------+-----+
|this coming tuesd...|  0.0|
|pin free dialing ...|  0.0|
|regards support team|  0.0|
|           thankskat|  0.0|
|speed dialing let...|  0.0|
|keep your user in...|  0.0|
| user name ilangostl|  0.0|
|now your family m...|  0.0|
|click on link bel...|  0.0|
|hi fellow little ...|  0.0|
|for every person ...|  0.0|
|we look forward t...|  0.0|
|thank you for cho...|  0.0|
|anbei die steuerb...|  0.0|
|we are having iss...|  0.0|
|re your message c...|  0.0|
|      hello rileystl|  0.0|
|thank you for usi...|  0.0|
|confirmation pending|  0.0|
|      angela baggett|  0.0|
+--------------------+-----+
only showing top 20 rows
*/

    lowerCasedDataFrame.printSchema
    /*  root
      |-- features: string (nullable = true)
      |-- label: double (nullable = false)*/

    lowerCasedDataFrame.columns
    //    //res23: Array[String] = Array(features, label)

    //Create a Tokenizer that according to the Spark API tokenizes ham and spam
    //sentences into individual lowercase words by whitespaces
    val mailTokenizer2 = new Tokenizer().setInputCol("lowerCasedSentences").setOutputCol("mailFeatureWords")
    //mailTokenizer2: org.apache.spark.ml.feature.Tokenizer = tok_0b4186779a55

    //The call to 'na' is meant for dropping any rows containing null values
    val naFunctions: DataFrameNaFunctions = lowerCasedDataFrame.na
    val nonNullBagOfWordsDataFrame = naFunctions.drop(Array("lowerCasedSentences"))

    println("Non-Null Bag Of lower-cased Words DataFrame looks like this:")
    nonNullBagOfWordsDataFrame.show()
    /*
    +--------------------+-----+
| lowerCasedSentences|label|
+--------------------+-----+
|this coming tuesd...|  0.0|
|pin free dialing ...|  0.0|
|regards support team|  0.0|
|           thankskat|  0.0|
|speed dialing let...|  0.0|
|keep your user in...|  0.0|
| user name ilangostl|  0.0|
|now your family m...|  0.0|
|click on link bel...|  0.0|
|hi fellow little ...|  0.0|
|for every person ...|  0.0|
|we look forward t...|  0.0|
|thank you for cho...|  0.0|
|anbei die steuerb...|  0.0|
|we are having iss...|  0.0|
|re your message c...|  0.0|
|      hello rileystl|  0.0|
|thank you for usi...|  0.0|
|confirmation pending|  0.0|
|      angela baggett|  0.0|
+--------------------+-----+
only showing top 20 rows
    */

    nonNullBagOfWordsDataFrame.columns
    //res38: Array[String] = Array(filteredMailFeatures, label, mailFeatureWords)

    nonNullBagOfWordsDataFrame.printSchema
    /*  root
 |-- lowerCasedSentences: string (nullable = true)
 |-- label: double (nullable = false)
 */

    val tokenizedBagOfWordsDataFrame: DataFrame = mailTokenizer2.transform(nonNullBagOfWordsDataFrame)

    println("Tokenized Non-Null Bag Of lower-cased Words DataFrame looks like this: ")
    tokenizedBagOfWordsDataFrame.show()
    /*
    +--------------------+-----+--------------------+
| lowerCasedSentences|label|    mailFeatureWords|
+--------------------+-----+--------------------+
|this coming tuesd...|  0.0|[this, coming, tu...|
|pin free dialing ...|  0.0|[pin, free, diali...|
|regards support team|  0.0|[regards, support...|
|           thankskat|  0.0|         [thankskat]|
|speed dialing let...|  0.0|[speed, dialing, ...|
|keep your user in...|  0.0|[keep, your, user...|
| user name ilangostl|  0.0|[user, name, ilan...|
|now your family m...|  0.0|[now, your, famil...|
|click on link bel...|  0.0|[click, on, link,...|
|hi fellow little ...|  0.0|[hi, fellow, litt...|
|for every person ...|  0.0|[for, every, pers...|
|we look forward t...|  0.0|[we, look, forwar...|
|thank you for cho...|  0.0|[thank, you, for,...|
|anbei die steuerb...|  0.0|[anbei, die, steu...|
|we are having iss...|  0.0|[we, are, having,...|
|re your message c...|  0.0|[re, your, messag...|
|      hello rileystl|  0.0|   [hello, rileystl]|
|thank you for usi...|  0.0|[thank, you, for,...|
|confirmation pending|  0.0|[confirmation, pe...|
|      angela baggett|  0.0|   [angela, baggett]|
+--------------------+-----+--------------------+
only showing top 20 rows

     */

    // StopWordRemover
    import org.apache.spark.ml.feature.StopWordsRemover

    //"features" is a "sentence"

    val stopWordRemover = new StopWordsRemover().setInputCol("mailFeatureWords").setOutputCol("noStopWordsMailFeatures") // same as "noStopWords"
    val noStopWordsDataFrame = stopWordRemover.transform(tokenizedBagOfWordsDataFrame)

    println("Tokenized Non-Null Bag Of lower-cased Words with no stopwords - this DataFrame looks like this:")
    noStopWordsDataFrame.show()
    /*
    +--------------------+-----+--------------------+-----------------------+
| lowerCasedSentences|label|    mailFeatureWords|noStopWordsMailFeatures|
+--------------------+-----+--------------------+-----------------------+
|this coming tuesd...|  0.0|[this, coming, tu...|   [coming, tuesday,...|
|pin free dialing ...|  0.0|[pin, free, diali...|   [pin, free, diali...|
|regards support team|  0.0|[regards, support...|   [regards, support...|
|           thankskat|  0.0|         [thankskat]|            [thankskat]|
|speed dialing let...|  0.0|[speed, dialing, ...|   [speed, dialing, ...|
|keep your user in...|  0.0|[keep, your, user...|   [keep, user, info...|
| user name ilangostl|  0.0|[user, name, ilan...|   [user, name, ilan...|
|now your family m...|  0.0|[now, your, famil...|   [family, member, ...|
|click on link bel...|  0.0|[click, on, link,...|   [click, link, ent...|
|hi fellow little ...|  0.0|[hi, fellow, litt...|   [hi, fellow, litt...|
|for every person ...|  0.0|[for, every, pers...|   [every, person, r...|
|we look forward t...|  0.0|[we, look, forwar...|   [look, forward, s...|
|thank you for cho...|  0.0|[thank, you, for,...|   [thank, choosing,...|
|anbei die steuerb...|  0.0|[anbei, die, steu...|   [anbei, die, steu...|
|we are having iss...|  0.0|[we, are, having,...|                [issue]|
|re your message c...|  0.0|[re, your, messag...|   [re, message, click]|
|      hello rileystl|  0.0|   [hello, rileystl]|      [hello, rileystl]|
|thank you for usi...|  0.0|[thank, you, for,...|   [thank, using, se...|
|confirmation pending|  0.0|[confirmation, pe...|   [confirmation, pe...|
|      angela baggett|  0.0|   [angela, baggett]|      [angela, baggett]|
+--------------------+-----+--------------------+-----------------------+
only showing top 20 rows
    */

    import session.implicits._

    ///**************Try this
    ////val noStopWordDataFrame2 = noStopWordDataFrame.select(explode($"filteredMailFeatures").alias("filteredMailFeatures"),noStopWordDataFrame("label"))

    val noStopWordsDataFrame2 = noStopWordsDataFrame.select(explode($"noStopWordsMailFeatures").alias("noStopWordsMailFeatures"),noStopWordsDataFrame("label"))

    println("Exploded: Tokenized Non-Null Bag Of lower-cased Words with no stopwords - this DataFrame looks like this: ")
    noStopWordsDataFrame2.show()

    ///*********************

    //for Hashing theory try this: http://openmymind.net/Back-To-Basics-Hashtables/ noStopWordsMailFeatures
    //************val hashMapper = new HashingTF().setInputCol(mailTokenizer2.getOutputCol).setOutputCol("mailFeatureVectors").setNumFeatures(10000)
    val hashMapper = new HashingTF().setInputCol("noStopWordsMailFeatures").setOutputCol("mailFeatureHashes").setNumFeatures(10000)
    //hashMapper: org.apache.spark.ml.feature.HashingTF = hashingTF_89eb55ea399c

    val featurizedDF1 = hashMapper.transform(noStopWordsDataFrame)

    println("Hash-Featurized AND Tokenized Non-Null Bag Of lower-cased Words with no stopwords - this DataFrame looks like this:")
    featurizedDF1.show()
    /*
    +--------------------+-----+--------------------+-----------------------+--------------------+
| lowerCasedSentences|label|    mailFeatureWords|noStopWordsMailFeatures|   mailFeatureHashes|
+--------------------+-----+--------------------+-----------------------+--------------------+
|this coming tuesd...|  0.0|[this, coming, tu...|   [coming, tuesday,...|(10000,[380,855,1...|
|pin free dialing ...|  0.0|[pin, free, diali...|   [pin, free, diali...|(10000,[1073,1097...|
|regards support team|  0.0|[regards, support...|   [regards, support...|(10000,[468,695,9...|
|           thankskat|  0.0|         [thankskat]|            [thankskat]|(10000,[5652],[1.0])|
|speed dialing let...|  0.0|[speed, dialing, ...|   [speed, dialing, ...|(10000,[1097,3245...|
|keep your user in...|  0.0|[keep, your, user...|   [keep, user, info...|(10000,[2904,5813...|
| user name ilangostl|  0.0|[user, name, ilan...|   [user, name, ilan...|(10000,[15,742,58...|
|now your family m...|  0.0|[now, your, famil...|   [family, member, ...|(10000,[1094,1181...|
|click on link bel...|  0.0|[click, on, link,...|   [click, link, ent...|(10000,[847,1719,...|
|hi fellow little ...|  0.0|[hi, fellow, litt...|   [hi, fellow, litt...|(10000,[1960,3391...|
|for every person ...|  0.0|[for, every, pers...|   [every, person, r...|(10000,[855,1073,...|
|we look forward t...|  0.0|[we, look, forwar...|   [look, forward, s...|(10000,[7923,9504...|
|thank you for cho...|  0.0|[thank, you, for,...|   [thank, choosing,...|(10000,[763,768,1...|
|anbei die steuerb...|  0.0|[anbei, die, steu...|   [anbei, die, steu...|(10000,[1409,1576...|
|we are having iss...|  0.0|[we, are, having,...|                [issue]|(10000,[6748],[1.0])|
|re your message c...|  0.0|[re, your, messag...|   [re, message, click]|(10000,[1719,2425...|
|      hello rileystl|  0.0|   [hello, rileystl]|      [hello, rileystl]|(10000,[1672,5048...|
|thank you for usi...|  0.0|[thank, you, for,...|   [thank, using, se...|(10000,[3322,3624...|
|confirmation pending|  0.0|[confirmation, pe...|   [confirmation, pe...|(10000,[1850,6339...|
|      angela baggett|  0.0|   [angela, baggett]|      [angela, baggett]|(10000,[6290,9622...|
+--------------------+-----+--------------------+-----------------------+--------------------+
only showing top 20 rows
    */

    //Create Training and test datasets
    //val splitDataSet =  featurizedDF1
    val splitFeaturizedDF1 = featurizedDF1.randomSplit(Array(0.80, 0.20), 98765L)
    //splitFeaturizedDF1: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([filteredMailFeatures: string, label: double ... 2 more fields],    [filteredMailFeatures: string, label: double ... 2 more fields])

    val testFeaturizedDF1 = splitFeaturizedDF1(1)
    println("TEST DATASET set count is: " + testFeaturizedDF1.count())

    val trainFeaturizedDF1 = splitFeaturizedDF1(0)
    println("TRAIN DATASET set count is: " + trainFeaturizedDF1.count())

    println("trainFeaturizedDF1 looks like this: ")
    trainFeaturizedDF1.show()

    val trainFeaturizedDF1New = trainFeaturizedDF1.drop("mailFeatureWords","noStopWordsMailFeatures","mailFeatureHashes")

    println("trainFeaturizedDF1 with 3 columns mailFeatureWords,noStopWordsMailFeatures,mailFeatureHashes dropped looks like this: ")
    trainFeaturizedDF1New.show()

    /*
      +--------------------+-----+
| lowerCasedSentences|label|
+--------------------+-----+
|pin free dialing ...|  0.0|
|regards support team|  0.0|
|this coming tuesd...|  0.0|
|speed dialing let...|  0.0|
| user name ilangostl|  0.0|
|for every person ...|  0.0|
|hi fellow little ...|  0.0|
|thank you for cho...|  0.0|
|we look forward t...|  0.0|
|anbei die steuerb...|  0.0|
|      angela baggett|  0.0|
|confirmation pending|  0.0|
|      hello rileystl|  0.0|
|              hi xxx|  0.0|
|    order nr baggett|  0.0|
|re your message c...|  0.0|
|thank you for usi...|  0.0|
|we are having iss...|  0.0|
|a parcel containi...|  0.0|
|                date|  0.0|
+--------------------+-----+
only showing top 20 rows

     */

    val mailIDF = new IDF().setInputCol("mailFeatureHashes").setOutputCol("mailIDF")
    val mailIDFFunction = mailIDF.fit(featurizedDF1)

    val normalizer = new Normalizer().setInputCol("mailIDF").setOutputCol("features")


    //Naive Bayes Algorithm

    val naiveBayes = new NaiveBayes().setFeaturesCol("features").setPredictionCol("prediction")

     val spamPipeline1 = new Pipeline().setStages(Array[PipelineStage](mailTokenizer2) ++
                                        Array[PipelineStage](stopWordRemover) ++
                                        Array[PipelineStage](hashMapper) ++
                                        Array[PipelineStage](mailIDF) ++
                                        Array[PipelineStage](normalizer) ++
                                        Array[PipelineStage](naiveBayes))
      //val spamPipeline1 = new Pipeline().setStages( Array(mailTokenizer2, stopWordRemover,hashMapper, mailIDF,normalizer, naiveBayes))

    //// Fit the pipeline to training documents.
    //***val spamModel1 = spamPipeline1.fit(trainFeaturizedDF1)
    val mailModel1 = spamPipeline1.fit(trainFeaturizedDF1New)

    //Make predictions on test dataset

     val rawPredictions = mailModel1.transform(testFeaturizedDF1.drop("mailFeatureWords","noStopWordsMailFeatures","mailFeatureHashes"))
     println("Predictions are: ")
     rawPredictions.show(100)
    /*

    Predictions are:
+--------------------+-----+--------------------+-----------------------+--------------------+--------------------+--------------------+--------------------+-----------+----------+
| lowerCasedSentences|label|    mailFeatureWords|noStopWordsMailFeatures|   mailFeatureHashes|             mailIDF|            features|       rawPrediction|probability|prediction|
+--------------------+-----+--------------------+-----------------------+--------------------+--------------------+--------------------+--------------------+-----------+----------+
|keep your user in...|  0.0|[keep, your, user...|   [keep, user, info...|(10000,[2904,5813...|(10000,[2904,5813...|(10000,[2904,5813...|[-25.886547708651...|      [1.0]|       0.0|
|           thankskat|  0.0|         [thankskat]|            [thankskat]|(10000,[5652],[1.0])|(10000,[5652],[3....|(10000,[5652],[1.0])|[-9.219476845326623]|      [1.0]|       0.0|
|click on link bel...|  0.0|[click, on, link,...|   [click, link, ent...|(10000,[847,1719,...|(10000,[847,1719,...|(10000,[847,1719,...|[-22.046815284741...|      [1.0]|       0.0|
|now your family m...|  0.0|[now, your, famil...|   [family, member, ...|(10000,[1094,1181...|(10000,[1094,1181...|(10000,[1094,1181...|[-28.536944937291...|      [1.0]|       0.0|
|das guthaben wird...|  0.0|[das, guthaben, w...|   [das, guthaben, w...|(10000,[568,2306,...|(10000,[568,2306,...|(10000,[568,2306,...|[-30.36554686638432]|      [1.0]|       0.0|
|      ideonecom team|  0.0|   [ideonecom, team]|      [ideonecom, team]|(10000,[468,5962]...|(10000,[468,5962]...|(10000,[468,5962]...|[-12.682918448753...|      [1.0]|       0.0|
| shipping contractor|  0.0|[shipping, contra...|   [shipping, contra...|(10000,[8273,9455...|(10000,[8273,9455...|(10000,[8273,9455...|[-12.839105392606...|      [1.0]|       0.0|
|     delivery status|  0.0|  [delivery, status]|     [delivery, status]|(10000,[7128,7497...|(10000,[7128,7497...|(10000,[7128,7497...|[-12.520898936789...|      [1.0]|       0.0|
|       use this link|  0.0|   [use, this, link]|            [use, link]|(10000,[4489,5066...|(10000,[4489,5066...|(10000,[4489,5066...|[-13.03830919264563]|      [1.0]|       0.0|
+--------------------+-----+--------------------+-----------------------+--------------------+--------------------+--------------------+--------------------+-----------+----------+
     */



    val predictions = rawPredictions.select($"lowerCasedSentences", $"prediction").cache
    println("Displaying Predictions as below:")
    predictions.show(50)


    session.stop()


  }




}