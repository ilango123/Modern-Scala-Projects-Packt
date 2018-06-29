package com.packt.modern.chapter4

import org.apache.spark.sql.SparkSession
import org.apache.logging.log4j.scala.Logging
import org.apache.logging.log4j.Level


trait SpamWrapper extends Logging{

  val hamSetFileName = "inbox2.txt"

  val spamFileName ="junk2.txt"

  //The entry point to programming Spark with the Dataset and DataFrame API.
  //This is the SparkSession

  lazy val session: SparkSession = {
    SparkSession
      .builder()
      .master("local")
      .appName("spam-classifier-pipeline")
      .getOrCreate()
  }

    //logger.getLogger("org").setLevel(Level.OFF)
  //logger.getLogger("akka").setLevel(Level.OFF)
    logger.apply(Level.OFF, "org")
    logger.apply(Level.OFF, "akka")

}























