import java.io.File

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object WordCount {

  def main(args: Array[String]): Unit = {
    countWord()
  }

  def countWord(): Unit = {
    // 创建Spark程序入口
    val sc = new SparkContext(getConfig())
    // 读文件，生成RDD
    val file: RDD[String] = sc.textFile("test.txt")
    // 按行分割 split=空格
    val words: RDD[String] = file.flatMap(_.split(" "))
    // 每个单词记一次，因为单词不区分大小写，这里用lambda表达式转为小写
    val wordOne: RDD[(String, Int)] = words.map(str => (str.toLowerCase(), 1))
    // reduce
//    val wordCount: RDD[(String, Int)] = wordOne.reduceByKey(_ + _)
        val wordCount: RDD[(String, Int)] = wordOne.reduceByKey((a, b) => (a + b))
    // 判断目标文件夹是否存在，存在则删除
    val target: File = new File("result")
    if (target.exists()) {
      println("delete!" + target.getAbsolutePath())
      deleteDir(target)
    }
    //保存
    wordCount.saveAsTextFile("result")
    sc.stop()
  }

  /**
    * 获取SparkConfig
    *
    * @return SparkConfig
    */
  def getConfig(): SparkConf = {
    val config: SparkConf = new SparkConf()
    config.setMaster("local")
    config.setAppName("WordCount")
    config.set("spark.hadoop.validateOutputSpecs", "false")
    return config
  }

  /**
    * 删除文件夹
    * @param dir
    */
  def deleteDir(dir: File): Unit = {
    if (dir.isFile) {
      dir.delete()
    } else {
      val files: Array[File] = dir.listFiles()
      if (files == null) {
        dir.delete()
      } else {
        for (i <- 0 to files.length-1) {
          deleteDir(files(i))
          dir.delete()
        }
      }
    }
  }

}
