"""
Zone C Vibration Threshold Detection & 3-Way Fault Diagnostic Engine
-------------------------------------------------------------------
This module evaluates machine telemetry and frequency spectrum data
against ISO 10816-7 / Europump Guidelines (2013) vibration severity standards.

When a machine's readings cross into Zone C (or exceed the configured threshold),
the engine classifies the root cause among the 3 distinct machinery problems:
  1. Imbalance (Unbalance)
  2. Misalignment
  3. High Frequency Vibration with High Temperature

ISO 10816-7 Zone C Thresholds (from Document Table 1, page 9):
  - Category 1 (<= 200 kW): Zone B upper limit = 4.0 mm/s, Zone C = 4.0 to 6.6 mm/s (Alarm: 5.0)
  - Category 1 (> 200 kW):  Zone B upper limit = 5.0 mm/s, Zone C = 5.0 to 7.6 mm/s (Alarm: 6.3)
  - Category 2 (<= 200 kW): Zone B upper limit = 5.1 mm/s, Zone C = 5.1 to 8.5 mm/s (Alarm: 6.4)
  - Category 2 (> 200 kW):  Zone B upper limit = 6.1 mm/s, Zone C = 6.1 to 9.5 mm/s (Alarm: 7.6)
  - Synthetic Dataset severe threshold: 7.0 mm/s (as configured in existing project code)

Temperature Threshold:
  - Normal operational limit: <= 65.0°C (clipping bound in synthetic dataset generators)
  - High temperature anomaly: > 65.0°C (bearing degradation, friction, lubrication failure)
"""

import os
import sys
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, DoubleType
)


# ISO 10816-7 Standard Threshold Table (mm/s RMS velocity on non-rotating parts)
ISO_10816_7_THRESHOLDS = {
    "Cat1_LE_200kW": {
        "zone_a_limit": 2.5,
        "zone_b_limit": 4.0,  # Boundary entering Zone C
        "zone_c_limit": 6.6,  # Boundary entering Zone D
        "alarm_limit": 5.0,
        "trip_limit": 8.3,
        "description": "Category 1 Pumps <= 200 kW (Critical / Hazardous)"
    },
    "Cat1_GT_200kW": {
        "zone_a_limit": 3.5,
        "zone_b_limit": 5.0,  # Boundary entering Zone C
        "zone_c_limit": 7.6,  # Boundary entering Zone D
        "alarm_limit": 6.3,
        "trip_limit": 9.5,
        "description": "Category 1 Pumps > 200 kW (Critical / Hazardous)"
    },
    "Cat2_LE_200kW": {
        "zone_a_limit": 3.2,
        "zone_b_limit": 5.1,  # Boundary entering Zone C
        "zone_c_limit": 8.5,  # Boundary entering Zone D
        "alarm_limit": 6.4,
        "trip_limit": 10.6,
        "description": "Category 2 Pumps <= 200 kW (General industrial)"
    },
    "Cat2_GT_200kW": {
        "zone_a_limit": 4.2,
        "zone_b_limit": 6.1,  # Boundary entering Zone C
        "zone_c_limit": 9.5,  # Boundary entering Zone D
        "alarm_limit": 7.6,
        "trip_limit": 11.9,
        "description": "Category 2 Pumps > 200 kW (General industrial)"
    },
    "Dataset_Default": {
        "zone_a_limit": 3.5,
        "zone_b_limit": 5.0,
        "zone_c_limit": 7.0,  # 7.0 mm/s calibrated threshold from project code
        "alarm_limit": 6.3,
        "trip_limit": 9.5,
        "description": "Project Calibrated Severe Zone C Threshold (7.0 mm/s)"
    }
}

TEMP_HIGH_THRESHOLD_C = 65.0  # Celsius


def get_spark_session(app_name="ZoneCDiagnostics"):
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


def diagnose_telemetry_dataset(spark, csv_path, standard_key="Dataset_Default", custom_threshold=7.0):
    """
    Evaluates telemetry data using PySpark to detect Zone C threshold crossings
    and diagnoses root causes (Imbalance, Misalignment, High Frequency Vibration with High Temperature).

    Parameters:
      spark: SparkSession
      csv_path: Path to cleaned telemetry CSV
      standard_key: Key in ISO_10816_7_THRESHOLDS
      custom_threshold: Numerical threshold to consider crossing into Zone C (default 7.0 mm/s)

    Returns:
      diagnosed_pandas_df: DataFrame with detected Zone C incidents and diagnosed problems
      summary_counts: Dictionary of diagnosed count per fault category
    """
    std_info = ISO_10816_7_THRESHOLDS.get(standard_key, ISO_10816_7_THRESHOLDS["Dataset_Default"])
    zone_c_threshold = custom_threshold if custom_threshold is not None else std_info["zone_b_limit"]

    print(f"\n{'='*75}")
    print(f"DIAGNOSTIC ANALYSIS FOR: {csv_path}")
    print(f"Applied Standard: {std_info['description']}")
    print(f"Zone C Trigger Threshold: {zone_c_threshold:.2f} mm/s RMS")
    print(f"High Temperature Threshold: {TEMP_HIGH_THRESHOLD_C:.1f} °C")
    print(f"{'='*75}")

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    total_records = df.count()

    locations = ["P1", "P2", "P3", "P4"]

    # Calculate max vibration per location: max(V, H, A)
    for loc in locations:
        df = df.withColumn(
            f"{loc}_max_vib",
            F.greatest(F.col(f"{loc}-V"), F.col(f"{loc}-H"), F.col(f"{loc}-A"))
        )

    # Calculate overall max vibration across all 4 locations
    df = df.withColumn(
        "max_machine_vibration",
        F.greatest(
            F.col("P1_max_vib"), F.col("P2_max_vib"),
            F.col("P3_max_vib"), F.col("P4_max_vib")
        )
    )

    # Calculate max temperature across all 4 locations
    df = df.withColumn(
        "max_machine_temp",
        F.greatest(
            F.col("P1-T"), F.col("P2-T"),
            F.col("P3-T"), F.col("P4-T")
        )
    )

    # Count how many bearing locations exceed the threshold
    zone_c_loc_count_expr = (
        F.when(F.col("P1_max_vib") >= zone_c_threshold, 1).otherwise(0) +
        F.when(F.col("P2_max_vib") >= zone_c_threshold, 1).otherwise(0) +
        F.when(F.col("P3_max_vib") >= zone_c_threshold, 1).otherwise(0) +
        F.when(F.col("P4_max_vib") >= zone_c_threshold, 1).otherwise(0)
    )
    df = df.withColumn("locations_in_zone_c_count", zone_c_loc_count_expr)

    # Filter to records where at least one bearing reading crossed into Zone C
    zone_c_df = df.filter(F.col("max_machine_vibration") >= zone_c_threshold)
    zone_c_incidents = zone_c_df.count()

    print(f"-> Total records evaluated: {total_records}")
    print(f"-> Records entering Zone C (vibration >= {zone_c_threshold:.2f} mm/s): {zone_c_incidents} ({zone_c_incidents/total_records:.2%})")

    if zone_c_incidents == 0:
        print("-> No machine readings crossed the Zone C threshold in this dataset.")
        return pd.DataFrame(), {}

    # Define PySpark diagnostic classification logic:
    # 1. High Frequency Vibration with High Temperature:
    #    - Max machine temperature exceeds TEMP_HIGH_THRESHOLD_C (65.0°C)
    # 2. Misalignment:
    #    - Normal temperature (<= 65.0°C)
    #    - Coordinated high readings across >= 3 locations simultaneously
    #    - Strong axial vibration component (Axial >= 75% of max radial at the primary high bearing)
    # 3. Imbalance:
    #    - Normal temperature (<= 65.0°C)
    #    - Primarily radial vibration (Axial < 75% of max radial)
    #    - Typically localized to pump/rotor bearing locations (P1/P2)

    # Helper columns to evaluate Axial vs Radial ratio at the dominant bearing
    for loc in locations:
        max_rad = F.greatest(F.col(f"{loc}-V"), F.col(f"{loc}-H"))
        axial = F.col(f"{loc}-A")
        df = df.withColumn(f"{loc}_axial_ratio", axial / F.when(max_rad > 0, max_rad).otherwise(1.0))

    # Determine dominant bearing location
    dominant_loc_expr = (
        F.when(F.col("max_machine_vibration") == F.col("P1_max_vib"), "P1")
        .when(F.col("max_machine_vibration") == F.col("P2_max_vib"), "P2")
        .when(F.col("max_machine_vibration") == F.col("P3_max_vib"), "P3")
        .otherwise("P4")
    )
    zone_c_df = zone_c_df.withColumn("dominant_location", dominant_loc_expr)

    # Dominant bearing axial ratio and temperature
    dom_axial_ratio_expr = (
        F.when(F.col("dominant_location") == "P1", F.col("P1-A") / F.greatest(F.col("P1-V"), F.col("P1-H")))
        .when(F.col("dominant_location") == "P2", F.col("P2-A") / F.greatest(F.col("P2-V"), F.col("P2-H")))
        .when(F.col("dominant_location") == "P3", F.col("P3-A") / F.greatest(F.col("P3-V"), F.col("P3-H")))
        .otherwise(F.col("P4-A") / F.greatest(F.col("P4-V"), F.col("P4-H")))
    )
    zone_c_df = zone_c_df.withColumn("dominant_axial_ratio", dom_axial_ratio_expr)

    # Diagnosis column
    diagnosis_expr = (
        F.when(
            F.col("max_machine_temp") > F.lit(TEMP_HIGH_THRESHOLD_C),
            F.lit("High Frequency Vibration with High Temperature")
        ).when(
            (F.col("locations_in_zone_c_count") >= 3) & (F.col("dominant_axial_ratio") >= 0.70),
            F.lit("Misalignment")
        ).otherwise(
            F.lit("Imbalance")
        )
    )

    zone_c_df = zone_c_df.withColumn("Diagnosed_Problem", diagnosis_expr)

    # ISO Zone classification column
    zone_classification_expr = (
        F.when(F.col("max_machine_vibration") > std_info["zone_c_limit"], "Zone D (Hazardous)")
        .otherwise("Zone C (Unsatisfactory / Limited Operation)")
    )
    zone_c_df = zone_c_df.withColumn("ISO_10816_7_Severity_Zone", zone_classification_expr)

    # Select key summary columns for output
    result_cols = [
        "Machine_number", "time stamp",
        "max_machine_vibration", "dominant_location",
        "locations_in_zone_c_count", "max_machine_temp",
        "ISO_10816_7_Severity_Zone", "Diagnosed_Problem"
    ]
    diagnosed_pdf = zone_c_df.select(result_cols).orderBy("time stamp", "Machine_number").toPandas()

    # Summary counts
    summary_counts = diagnosed_pdf["Diagnosed_Problem"].value_counts().to_dict()
    print("\n-> Root Cause Breakdown of Zone C Exceedances:")
    for prob, cnt in summary_counts.items():
        print(f"   * {prob}: {cnt} incidents ({cnt / zone_c_incidents:.1%})")

    return diagnosed_pdf, summary_counts


def diagnose_frequency_spectrum(spectrum_csv_path, threshold=7.0):
    """
    Evaluates frequency spectrum datasets (e.g. pump_imbalance_frequency_spectrum.csv,
    pump_high_frequency_spectrum.csv) to diagnose Zone C crossings.

    Distinguishes:
      1. Imbalance: 1x running frequency amplitude accounts for > 70% of velocity energy.
      2. Misalignment: Prominent 2x running frequency harmonic (2x amplitude >= 40% of 1x).
      3. High Frequency Vibration with High Temperature: High-frequency harmonics (orders > 3x,
         bearing defect orders BPFO/BPFI, 200-500 Hz) dominate and/or bearing temperature > 65°C.
    """
    print(f"\n{'='*75}")
    print(f"SPECTRAL DIAGNOSTIC ANALYSIS FOR: {spectrum_csv_path}")
    print(f"{'='*75}")

    df = pd.read_csv(spectrum_csv_path)
    machine = df["Machine_number"].iloc[0]
    rpm = df["RPM"].iloc[0]
    run_freq = df["Running_frequency_Hz"].iloc[0]

    # Evaluate overall RMS per (Location, Direction)
    loc_dir_groups = df.groupby(["Location", "Direction"])
    summary_rows = []

    for (loc, direction), group in loc_dir_groups:
        overall_rms = group["Overall_RMS_mm_s"].iloc[0]
        temp = group["Bearing_Temperature_C"].iloc[0] if "Bearing_Temperature_C" in group.columns else 45.0

        in_zone_c = overall_rms >= threshold

        # Energy distribution across harmonic orders
        # 1x running frequency: orders 0.95 to 1.05
        # 2x running frequency: orders 1.95 to 2.05
        # 3x running frequency: orders 2.95 to 3.05
        # High frequency / high orders: orders > 3.2
        group_orders = group.set_index("Frequency_Order")["Velocity_RMS_mm_s"]
        
        # Interpolate/extract nearest order amplitudes
        def get_amp(order_target, tol=0.1):
            subset = group[np.abs(group["Frequency_Order"] - order_target) <= tol]
            return subset["Velocity_RMS_mm_s"].max() if not subset.empty else 0.0

        amp_1x = get_amp(1.0)
        amp_2x = get_amp(2.0)
        amp_high_freq = group[group["Frequency_Order"] >= 3.2]["Velocity_RMS_mm_s"].max()

        # Diagnosis logic for spectrum
        if not in_zone_c:
            diagnosis = "Normal / Acceptable (Zone A/B)"
        elif temp > TEMP_HIGH_THRESHOLD_C or amp_high_freq > amp_1x * 0.8:
            diagnosis = "High Frequency Vibration with High Temperature"
        elif amp_2x >= amp_1x * 0.4:
            diagnosis = "Misalignment"
        elif amp_1x >= overall_rms * 0.65 or amp_1x > amp_2x * 2.0:
            diagnosis = "Imbalance"
        else:
            diagnosis = "Unspecified High Vibration"

        summary_rows.append({
            "Machine_number": machine,
            "Location": loc,
            "Direction": direction,
            "Overall_RMS_mm_s": overall_rms,
            "Temperature_C": temp,
            "1x_Amp_mm_s": round(amp_1x, 3),
            "2x_Amp_mm_s": round(amp_2x, 3),
            "High_Freq_Max_Amp": round(amp_high_freq, 3),
            "In_Zone_C": in_zone_c,
            "Diagnosed_Problem": diagnosis
        })

    spectrum_res_df = pd.DataFrame(summary_rows)
    print(spectrum_res_df[["Location", "Direction", "Overall_RMS_mm_s", "Temperature_C", "In_Zone_C", "Diagnosed_Problem"]])
    return spectrum_res_df


def main():
    spark = get_spark_session()
    try:
        print("\n" + "#"*75)
        print("EXECUTING ZONE C ROOT CAUSE DIAGNOSTICS PIPELINE")
        print("#"*75)

        # 1. Evaluate Misalignment Cleaned Dataset
        res_misalign, counts_misalign = diagnose_telemetry_dataset(
            spark,
            "cleaned_misalignment_dataset_2024_2025.csv",
            standard_key="Dataset_Default",
            custom_threshold=7.0
        )
        if not res_misalign.empty:
            res_misalign.to_csv("diagnosed_misalignment_events.csv", index=False)
            print("Sample diagnosed events:")
            print(res_misalign.head(5))

        # 2. Evaluate High Frequency + High Temperature Cleaned Dataset
        res_hf_ht, counts_hf_ht = diagnose_telemetry_dataset(
            spark,
            "cleaned_high_freq_high_temp_dataset_2024_2025.csv",
            standard_key="Dataset_Default",
            custom_threshold=7.0
        )
        if not res_hf_ht.empty:
            res_hf_ht.to_csv("diagnosed_high_freq_high_temp_events.csv", index=False)
            print("Sample diagnosed events:")
            print(res_hf_ht.head(5))

        # 3. Evaluate Daily Baseline Cleaned Dataset
        res_daily, counts_daily = diagnose_telemetry_dataset(
            spark,
            "cleaned_pump_motor_vibration_2024_2025_daily.csv",
            standard_key="Dataset_Default",
            custom_threshold=7.0
        )
        if not res_daily.empty:
            res_daily.to_csv("diagnosed_daily_baseline_events.csv", index=False)
            print("Sample diagnosed events:")
            print(res_daily.head(5))

        # 4. Evaluate Frequency Spectra (Imbalance & High Frequency)
        print("\n" + "-"*75)
        print("FREQUENCY SPECTRUM DIAGNOSTICS")
        print("-"*75)
        spec_imbalance = diagnose_frequency_spectrum("pump_imbalance_frequency_spectrum.csv", threshold=7.0)
        spec_imbalance.to_csv("diagnosed_spectrum_imbalance.csv", index=False)

        spec_hf = diagnose_frequency_spectrum("pump_high_frequency_spectrum.csv", threshold=7.0)
        spec_hf.to_csv("diagnosed_spectrum_high_freq.csv", index=False)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
