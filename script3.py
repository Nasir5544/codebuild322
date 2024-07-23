import sys
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# from pyspark.sql.functions import (to_timestamp, to_date, hour, collect_list, size, lit, udf, explode, array,
#                                  countDistinct, count, current_timestamp, year, month, dayofmonth, round as py_round)
import pyspark.sql.functions as F
from pyspark.sql.types import (DoubleType, StructType, ArrayType, StructField, Row, IntegerType, DataType, StringType,
                               LongType)

from src.common.helpers.cw_logging_helper import CloudWatchLogsManager
from src.common.helpers.parameters_helper import ParametersHelper
from src.common.helpers.athena_helper import AthenaHelper
from src.common.helpers.sns_helper import SNSHelper
from src.common.helpers.s3_helper import S3Helper
from src.common.utils.trip_data_processor import TripDataProcessor
from src.common.utils.speed_filter_detection import SpeedFilterDetection

udf_old_formula_result_schema = DoubleType()

trip_processor = TripDataProcessor()
speed_filter_detection = SpeedFilterDetection()


class MonthlyStatJob:
    """
        Class for the Carrot driving_monthly_stat Job
    """

    def __init__(self, spark_session, load_type="INCREMENTAL") -> None:
        """
            Initializer for driving_monthly_stat class object
        """
        self.parameter_helper = ParametersHelper()
        self.parameters = self.parameter_helper.parameters
        self.athena_helper = AthenaHelper(region_name=self.parameters.region)
        self.cw_manager = CloudWatchLogsManager(
            log_group_name=self.parameters.cw_log_group_name_gold,
            region_name=self.parameters.region)

        self.sns_helper = SNSHelper(region_name=self.parameters.region)
        self.s3_helper = S3Helper(region_name=self.parameters.region)

        self.spark_session = spark_session
        self.load_type = load_type

        self.cw_manager = CloudWatchLogsManager(
            log_group_name=self.parameters.cw_log_group_name_gold,
            region_name=self.parameters.region)

    def _check_target_catalog_and_data_exist(self, bucket_name, db_name, table_name):

        # check athena catalog exists
        catalog_exists = self.athena_helper.check_catalog_exists(bucket_name, db_name, table_name)
        # check parquet files on s3 exists
        data_exists = self.s3_helper.check_table_dir_exists(bucket_name, table_name)

        return True if (catalog_exists and data_exists) else False

    def _extract_new_partition_cols(self, dataframe):
        distinct_part_cols = None
        partition_paths = []
        if not dataframe.rdd.isEmpty():
            # Get distinct new partition values in the new source data
            df_distinct = dataframe.select(dataframe.year, dataframe.month).distinct()
            distinct_part_cols = df_distinct.collect()

            for part in distinct_part_cols:
                partition_paths.append(
                    f"year={part['year']}/month={part['month']}/")

        return partition_paths

    def _read_target_latest_part_cols(self, bucket, prefix):
        try:
            # Construct the S3 path
            s3_path = f's3://{bucket}/{prefix}'
            df = self.spark_session.read.parquet(s3_path)
            df.createOrReplaceTempView("monthly_stat_df_tmp")

            query = """
                select 
                    max(year) as year, 
                    max(month) as month
                from monthly_stat_df_tmp 
                group by year, month
                order by year, month asc;
            """

            df = self.spark_session.sql(query)
            last_row = df.tail(1)[0]
            return last_row
        except Exception as error:
            print("Target table does not exist: ", error)
            raise error

    def extract_latest_part_cols(self):
        latest_source_part_cols = self._read_target_latest_part_cols(
            self.parameters.gold_data_bucket_path, self.parameters.driving_monthly_stat_gold_table_name
        )

        year = int(latest_source_part_cols["year"])
        month = int(latest_source_part_cols["month"])

        return datetime(year, month, 1)

    def _read_data_from_s3(self, bucket_name, table_name, target_partition_datetime):
        # Construct the S3 path
        s3_path = f's3://{bucket_name}/{table_name}'

        df = self.spark_session.read.parquet(s3_path)

        if self.load_type == "INCREMENTAL":
            # If load type is set to be INCREMENTAL, then filter data accordingly
            df = df.withColumn("partition_datetime", F.to_timestamp(
                F.concat(df.year, F.lit("-"), df.month, F.lit("-"), F.lit(1), F.lit(" "), F.lit("00"), F.lit(":"),
                         F.lit("00"), F.lit(":"), F.lit("00")), 'yyyy-MM-dd HH:mm:ss'))

            df = df.filter(df.partition_datetime > target_partition_datetime)
        return df

    def _write_data_to_s3(self, df, buket_name, table_name):
        write_mode = "overwrite"
        if self.load_type == "INCREMENTAL":
            write_mode = "append"

        df.write.format("parquet").mode(write_mode) \
            .partitionBy("year", "month") \
            .save(f"s3://{buket_name}/{table_name}")

    @staticmethod
    @F.udf(udf_old_formula_result_schema)
    def apply_distance_formula(latitude_positions, longitude_positions, size_of_lat):
        try:
            return trip_processor.calculate_total_distance(latitude_positions, longitude_positions, size_of_lat)
        except Exception as error:
            raise error

    def refresh_monthly_stat_table(self):
        """
            Main driver method for driving_monthly_stat table for Gold Layer
        """
        current_datetime = datetime.now()
        try:
            self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                    str(current_datetime.month),
                                                                                    str(current_datetime.day),
                                                                                    str(current_datetime.hour),
                                                                                    str(current_datetime.minute)]),
                                     f"Started Gold Layer Job for "
                                     f"{self.parameters.driving_monthly_stat_gold_table_name}"
                                     f" table with params {self.parameters} with load type: {self.load_type}")

            # check target catalog and s3 data exists
            history_exists = self._check_target_catalog_and_data_exist(
                self.parameters.gold_data_bucket_path, self.parameters.GLUE_DATABASE_GOLD_NAME,
                self.parameters.driving_monthly_stat_gold_table_name)

            self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                    str(current_datetime.month),
                                                                                    str(current_datetime.day),
                                                                                    str(current_datetime.hour),
                                                                                    str(current_datetime.minute)]),
                                     f"History status check result: {history_exists}")

            if self.load_type == "INCREMENTAL" and not history_exists:
                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"No history found for the "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} table on Athena OR "
                                         f"S3. Going to switch LOAD TYPE to FULL_LOAD")
                self.load_type = "FULL_LOAD"

            target_part_datetime = None
            if self.load_type == "INCREMENTAL":
                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Started fetching value for partition columns for the latest data available "
                                         f"in {self.parameters.driving_monthly_stat_gold_table_name} table")

                target_part_datetime = self.extract_latest_part_cols()

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Completed fetching value for partition columns for the latest data available"
                                         f" in {self.parameters.driving_monthly_stat_gold_table_name} table")

            self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                    str(current_datetime.month),
                                                                                    str(current_datetime.day),
                                                                                    str(current_datetime.hour),
                                                                                    str(current_datetime.minute)]),
                                     f"Started reading data from S3 for Gold Layer Job for "
                                     f"{self.parameters.driving_position_silver_table_name}.")
            # Read driving_summary and driving_position data from silver bucket
            position_df = self._read_data_from_s3(
                self.parameters.silver_data_bucket_path,
                self.parameters.driving_position_silver_table_name, target_part_datetime)

            self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                    str(current_datetime.month),
                                                                                    str(current_datetime.day),
                                                                                    str(current_datetime.hour),
                                                                                    str(current_datetime.minute)]),
                                     f"Completed reading data from S3 for Gold Layer Job for "
                                     f"{self.parameters.driving_position_silver_table_name} table.")

            if not position_df.rdd.isEmpty():
                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Discovered new data available on source that needs to e processed for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} "
                                         f"table for Gold Layer.")

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Started preparing data for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} "
                                         f"table for Gold Layer.")
                monthly_stat_df = position_df.select(
                    position_df.trip_id,
                    position_df.user_id,
                    position_df.vehicle_id,
                    position_df.ts,
                    position_df.latitude,
                    position_df.longitude,
                    position_df.ct,
                    position_df.sp
                )

                monthly_stat_df = monthly_stat_df.withColumn(
                    'ts', F.to_timestamp(monthly_stat_df.ts, 'yyyy-MM-dd HH:mm:ss'))
                monthly_stat_df = monthly_stat_df.withColumn('date', F.to_date(monthly_stat_df.ts))
                monthly_stat_df = monthly_stat_df.withColumn('month', F.month(monthly_stat_df.ts))

                # Group By on date and day for monthly stats
                # Perform following aggregates:
                #   - count unique trip ids to get total trip count
                #   - count unique user ids to get total user count
                #   - count unique vehicle ids to get total vehicle count
                #   - count all ct values to get all ct entries for all trips recorded under positions per day group
                #   - combine all latitude as list per day group
                #   - combine all longitude as list per day group
                #   - combine all ct as list per day group
                #   - combine all sp as list per day group
                agg_monthly_stat_df = monthly_stat_df.orderBy(
                    monthly_stat_df.date, monthly_stat_df.month, monthly_stat_df.trip_id, monthly_stat_df.ct)\
                    .groupBy(monthly_stat_df.date, monthly_stat_df.month, monthly_stat_df.trip_id).agg(
                        F.countDistinct("user_id").alias("user_count"),
                        F.countDistinct("vehicle_id").alias("vehicle_count"),
                        F.count("ct").alias("total_driving_time"),
                        F.collect_list("latitude").alias("latitude"),
                        F.collect_list("longitude").alias("longitude")
                    )

                agg_monthly_stat_df = agg_monthly_stat_df.withColumn(
                    "size_of_latitude", F.size(agg_monthly_stat_df.latitude))

                agg_monthly_stat_df = agg_monthly_stat_df.withColumn("total_distance", self.apply_distance_formula(
                    agg_monthly_stat_df.latitude,
                    agg_monthly_stat_df.longitude,
                    agg_monthly_stat_df.size_of_latitude
                ))

                agg_monthly_stat_df = agg_monthly_stat_df.withColumn(
                    "total_distance", F.round(agg_monthly_stat_df.total_distance).cast('long'))

                new_agg_monthly_stat_df = agg_monthly_stat_df.groupBy(
                    agg_monthly_stat_df.date, agg_monthly_stat_df.month).agg(
                    F.countDistinct("trip_id").alias("trip_count"),
                    F.sum("user_count").alias("user_count"),
                    F.sum("vehicle_count").alias("vehicle_count"),
                    F.sum(agg_monthly_stat_df.total_driving_time).alias("total_driving_time"),
                    F.sum(agg_monthly_stat_df.total_distance).alias("total_distance")
                )

                new_agg_monthly_stat_df = new_agg_monthly_stat_df.withColumn(
                    'total_driving_time', F.round(new_agg_monthly_stat_df.total_driving_time / 60).cast('long'))

                # Add remaining partition columns to the DataFrame
                agg_df_with_partition_cols = \
                    new_agg_monthly_stat_df.withColumn("year", F.year("date"))

                agg_df_with_partition_cols = agg_df_with_partition_cols.withColumn(
                    "year_month", F.concat(agg_df_with_partition_cols.year, agg_df_with_partition_cols.month))

                final_monthly_stat_df = agg_df_with_partition_cols.select(
                    "year_month", "month",  "trip_count", "user_count", "vehicle_count", "total_driving_time",
                    "total_distance", "year")

                final_monthly_stat_df = final_monthly_stat_df.withColumn(
                    "avg_distance", final_monthly_stat_df.total_distance / final_monthly_stat_df.trip_count)

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Completed preparing data for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} "
                                         f"table for Gold Layer.")

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Started writing data to S3 for Gold Layer Job for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} table.")
                # # write data to s3 and update catalog
                self._write_data_to_s3(final_monthly_stat_df, self.parameters.gold_data_bucket_path,
                                       self.parameters.driving_monthly_stat_gold_table_name)

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Completed writing data to S3 for Gold Layer Job for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} table.")

                partition_paths = self._extract_new_partition_cols(final_monthly_stat_df)

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                       str(current_datetime.month),
                                                                                       str(current_datetime.day),
                                                                                       str(current_datetime.hour),
                                                                                       str(current_datetime.minute)]),
                                         f"Creating/Updating Catalog for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} data on ATHENA "
                                         f"started")

                self.athena_helper.catalog_driving_monthly_stat_table(
                    self.parameters.gold_data_bucket_path, partition_paths, self.parameters.glue_database_gold_name,
                    self.parameters.driving_monthly_stat_gold_table_name, self.load_type)

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                       str(current_datetime.month),
                                                                                       str(current_datetime.day),
                                                                                       str(current_datetime.hour),
                                                                                       str(current_datetime.minute)]),
                                         f"Creating/Updating Catalog for ATHENA completed successfully "
                                         f"for database name: {self.parameters.glue_database_gold_name} and "
                                         f"table name: {self.parameters.driving_monthly_stat_gold_table_name}")

                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"Completed Gold Layer Job for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} table.")
            else:
                self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                        str(current_datetime.month),
                                                                                        str(current_datetime.day),
                                                                                        str(current_datetime.hour),
                                                                                        str(current_datetime.minute)]),
                                         f"No new unprocessed data available on source for Gold Layer Job for "
                                         f"{self.parameters.driving_monthly_stat_gold_table_name} table.")

        except Exception as error:
            self.cw_manager.log_error(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                     str(current_datetime.month),
                                                                                     str(current_datetime.day),
                                                                                     str(current_datetime.hour),
                                                                                     str(current_datetime.minute)]),
                                      f"Failed Gold Layer Job for "
                                      f"{self.parameters.driving_monthly_stat_gold_table_name} table.")

            self.cw_manager.log_error(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                     str(current_datetime.month),
                                                                                     str(current_datetime.day),
                                                                                     str(current_datetime.hour),
                                                                                     str(current_datetime.minute)]),
                                      f"Following error occurred during Gold Layer Job for "
                                      f"{self.parameters.driving_monthly_stat_gold_table_name} table: "
                                      f"{error}", exception=error)

            # Send SNS Notification for failure
            topic_arn = self.parameters.sns_topic_failure
            sns_subject = f"GOLD Layer Job Failed for Table: {self.parameters.driving_monthly_stat_gold_table_name} " \
                          f"| Carrot Datalake"
            sns_message = "This SNS is to notify that the Batch ETL Job from Carrot Data lake for GOLD Layer's " \
                          f"{self.parameters.driving_monthly_stat_gold_table_name} table has FAILED. \n" \
                          f"Find the Cause of Failure and Stack Trace below: \n" \
                          f"{error}"
            self.sns_helper.publish_notification(topic_arn=topic_arn, subject=sns_subject, message=sns_message)
            self.cw_manager.log_info(self.parameters.CW_LOG_STREAM_GOLD + '/'.join([str(current_datetime.year),
                                                                                    str(current_datetime.month),
                                                                                    str(current_datetime.day),
                                                                                    str(current_datetime.hour),
                                                                                    str(current_datetime.minute)]),
                                     f"Failure SNS Notification successfully sent to "
                                     f"Topic: {topic_arn.split(':')[-1]} with Subject: "
                                     f"{sns_subject} and Message: {sns_message}")
            raise error


if __name__ == "__main__":
    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)

    # @params: [JOB_NAME]
    args = getResolvedOptions(sys.argv, ['JOB_NAME', 'LOAD_TYPE'])

    job.init(args['JOB_NAME'], args)
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    arg_load_type = args.get('LOAD_TYPE', 'INCREMENTAL')

    if args.get('LOAD_TYPE', 'INCREMENTAL') in ['FULL_LOAD', 'INCREMENTAL']:
        stream_driver = MonthlyStatJob(spark, load_type=arg_load_type)
        stream_driver.refresh_monthly_stat_table()
    else:
        print(f"Invalid value: {arg_load_type} provided for arg --LOAD_TYPE")
        print("Exiting Job by raising exception")
        raise ValueError(f"Invalid value: {arg_load_type} provided for arg --LOAD_TYPE")
    job.commit()
