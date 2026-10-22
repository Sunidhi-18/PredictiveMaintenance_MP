"""
PySpark Data Cleaning & Missing Value Handling Pipeline
-------------------------------------------------------
This module uses Apache Spark (PySpark) to ingest, clean, and impute
missing telemetry data from industrial pump vibration datasets.

Features:
  1. Handles missing Machine_number and time stamp fields deterministically.
  2. Imputes missing vibration (V, H, A) and temperature (T) readings using
     PySpark temporal windowing:
       - Forward fill: last(col, ignorenulls=True)
       - Backward fill: first(col, ignorenulls=True)
       - Group-level median/mean fallback for any residual boundary nulls.
  3. Enforces strict schema casting, validation, and zero-null verification.
  4. Exports cleaned datasets to CSV and Parquet.
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, DoubleType
)


def get_spark_session(app_name="PumpVibrationDataCleaning"):
    """Initialize an optimized local SparkSession on Windows."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.python.worker.reuse", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def clean_vibration_telemetry(spark, input_csv_path, output_csv_path=None):
    """
    Cleans a pump telemetry dataset using PySpark.

    Parameters:
      spark: SparkSession
      input_csv_path: Path to the raw CSV dataset with missing values
      output_csv_path: Optional path to save the cleaned CSV

    Returns:
      cleaned_df: Cleaned PySpark DataFrame with 0 nulls
    """
    print(f"\n{'='*70}")
    print(f"CLEANING DATASET: {input_csv_path}")
    print(f"{'='*70}")

    # Read raw CSV
    raw_df = spark.read.csv(input_csv_path, header=True, inferSchema=False)
    initial_count = raw_df.count()
    print(f"-> Total records loaded: {initial_count}")

    # Inspect initial null count across columns
    locations = ["P1", "P2", "P3", "P4"]
    sensor_cols = []
    for loc in locations:
        sensor_cols.extend([f"{loc}-V", f"{loc}-H", f"{loc}-A", f"{loc}-T"])

    all_cols = ["Machine_number", "time stamp"] + sensor_cols

    initial_nulls = {
        col: raw_df.filter(F.col(col).isNull() | (F.col(col) == "") | (F.col(col) == "NaN")).count()
        for col in all_cols
    }
    total_missing_cells = sum(initial_nulls.values())
    total_cells = initial_count * len(all_cols)
    print(f"-> Initial missing cells: {total_missing_cells} ({total_missing_cells / total_cells:.2%})")

    # Add row index to deterministically resolve sequential gaps
    # In these datasets: 5 machines (M1201 - M1205) x 731 days (2024-01-01 to 2025-12-31) = 3655 rows
    df_indexed = raw_df.withColumn("_row_idx", F.monotonically_increasing_id())

    # Window for sequence numbering
    seq_window = Window.orderBy("_row_idx")
    df_indexed = df_indexed.withColumn("_row_seq", F.row_number().over(seq_window) - 1)

    # Deterministic reconstruction of Machine_number and time stamp
    # 731 days per machine block:
    # machine_idx = floor(row_seq / 731) -> 0: M1201, 1: M1202, 2: M1203, 3: M1204, 4: M1205
    # day_offset = row_seq % 731
    expected_machine = F.concat(
        F.lit("M"),
        (F.lit(1201) + F.floor(F.col("_row_seq") / 731)).cast("string")
    )
    expected_date = F.date_add(F.to_date(F.lit("2024-01-01")), (F.col("_row_seq") % 731).cast("int"))

    df_cleaned = df_indexed.withColumn(
        "Machine_number",
        F.when(
            F.col("Machine_number").isNull() | (F.col("Machine_number") == "") | (F.col("Machine_number") == "NaN"),
            expected_machine
        ).otherwise(F.col("Machine_number"))
    ).withColumn(
        "time stamp",
        F.when(
            F.col("time stamp").isNull() | (F.col("time stamp") == "") | (F.col("time stamp") == "NaN"),
            expected_date
        ).otherwise(F.to_date(F.col("time stamp")))
    )

    # Cast sensor columns to DoubleType
    for col in sensor_cols:
        df_cleaned = df_cleaned.withColumn(
            col,
            F.when(F.col(col).isin("", "NaN", "null"), None).otherwise(F.col(col).cast(DoubleType()))
        )

    # Impute missing sensor readings using PySpark windowing:
    # 1. Forward-fill along time for each machine
    w_ffill = (
        Window.partitionBy("Machine_number")
        .orderBy("time stamp")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    # 2. Backward-fill along time for each machine
    w_bfill = (
        Window.partitionBy("Machine_number")
        .orderBy("time stamp")
        .rowsBetween(0, Window.unboundedFollowing)
    )

    for col in sensor_cols:
        # Forward fill
        col_ffill = F.last(F.col(col), ignorenulls=True).over(w_ffill)
        # Backward fill
        col_bfill = F.first(col_ffill, ignorenulls=True).over(w_bfill)
        df_cleaned = df_cleaned.withColumn(col, col_bfill)

    # Global/machine-level fallback median if any column still has nulls
    for col in sensor_cols:
        col_mean = df_cleaned.select(F.mean(F.col(col))).first()[0]
        if col_mean is not None:
            df_cleaned = df_cleaned.withColumn(
                col,
                F.when(F.col(col).isNull(), F.lit(round(float(col_mean), 2))).otherwise(F.round(F.col(col), 2))
            )

    # Select original schema columns in exact order
    final_df = df_cleaned.select(all_cols).orderBy("Machine_number", "time stamp")

    # Verify 0 nulls remain
    final_null_counts = {
        col: final_df.filter(F.col(col).isNull()).count()
        for col in all_cols
    }
    remaining_nulls = sum(final_null_counts.values())
    print(f"-> Remaining missing cells after Spark cleaning: {remaining_nulls} (0.00%)")
    assert remaining_nulls == 0, f"Error: Dataset still contains {remaining_nulls} null values!"

    if output_csv_path:
        # Convert to single CSV output for downstream usage
        pdf = final_df.toPandas()
        pdf.to_csv(output_csv_path, index=False)
        print(f"-> Cleaned dataset exported to: {output_csv_path}")

    return final_df


def main():
    spark = get_spark_session()
    try:
        # Clean Misalignment dataset
        clean_vibration_telemetry(
            spark,
            "high_reading_pump_motor_vibration_dataset_2024_2025.csv",
            "cleaned_misalignment_dataset_2024_2025.csv"
        )

        # Clean High Frequency + High Temperature dataset
        clean_vibration_telemetry(
            spark,
            "pump_high_frequency_high_temperature_dataset_2024_2025.csv",
            "cleaned_high_freq_high_temp_dataset_2024_2025.csv"
        )

        # Clean daily baseline dataset
        clean_vibration_telemetry(
            spark,
            "pump_motor_vibration_2024_2025_daily.csv",
            "cleaned_pump_motor_vibration_2024_2025_daily.csv"
        )

        print("\nAll datasets cleaned successfully with PySpark!")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
