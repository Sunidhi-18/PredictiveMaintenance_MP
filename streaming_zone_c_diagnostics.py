"""
Real-Time Streaming Zone C Detection & 3-Way Fault Diagnostic Engine
-------------------------------------------------------------------
This module transforms static pump datasets into continuous streaming data:
  - Ingests data 1 row at a time (1 row = 1 streaming telemetry packet).
  - Handles missing values on-the-fly using stateful streaming imputation
    (maintains running machine state for missing IDs, timestamps, and sensor readings).
  - Evaluates each incoming streaming event against ISO 10816-7 Zone C threshold (7.0 mm/s)
    and temperature threshold (65.0°C).
  - Immediately classifies any Zone C breach into one of the 3 problems:
      1. Imbalance
      2. Misalignment
      3. High Frequency Vibration with High Temperature
  - Emits real-time streaming alerts and logs to stdout and output sink.
"""

import os
import sys
import time
import math
import pandas as pd
import numpy as np

# Threshold configurations from ISO 10816-7 / Europump Guidelines
ZONE_C_THRESHOLD_MM_S = 7.0   # mm/s RMS velocity (Zone C trigger)
TEMP_HIGH_THRESHOLD_C = 65.0  # °C (Bearing thermal warning)

LOCATIONS = ["P1", "P2", "P3", "P4"]
DIRECTIONS = ["V", "H", "A"]


class StreamingPumpProcessor:
    """
    Stateful Real-Time Streaming Processor for Industrial Pump Telemetry.
    Processes 1 row at a time, maintains machine telemetry state, imputes missing
    values dynamically, and classifies Zone C faults in real time.
    """

    def __init__(self, zone_c_threshold=ZONE_C_THRESHOLD_MM_S, temp_threshold=TEMP_HIGH_THRESHOLD_C):
        self.zone_c_threshold = zone_c_threshold
        self.temp_threshold = temp_threshold

        # Stateful cache per machine: tracks last valid sensor readings and timestamp
        self.machine_state = {}
        self.last_known_machine = "M1201"

        # Streaming metrics
        self.total_stream_count = 0
        self.imputed_fields_count = 0
        self.zone_c_alert_count = 0
        self.fault_breakdown = {
            "Imbalance": 0,
            "Misalignment": 0,
            "High Frequency Vibration with High Temperature": 0,
        }

    def process_stream_event(self, raw_row, row_index=0):
        """
        Processes a single incoming streaming row.

        Parameters:
          raw_row: dict representing 1 row of streaming telemetry
          row_index: sequence index of the stream event

        Returns:
          event_result: dict with cleaned values, Zone C status, and diagnosis
        """
        self.total_stream_count += 1
        cleaned_row = {}

        # ----------------------------------------------------------------------
        # 1. Stateful Imputation of Machine Number & Timestamp
        # ----------------------------------------------------------------------
        raw_machine = raw_row.get("Machine_number")
        if pd.isna(raw_machine) or str(raw_machine).strip() in ("", "nan", "None", "NULL"):
            # Infer from stream block (731 rows per machine) or last known machine
            inferred_machine_num = 1201 + (row_index // 731)
            machine = f"M{inferred_machine_num}"
            self.imputed_fields_count += 1
        else:
            machine = str(raw_machine).strip()
            self.last_known_machine = machine

        cleaned_row["Machine_number"] = machine

        # Initialize machine state if not present
        if machine not in self.machine_state:
            self.machine_state[machine] = {
                "last_timestamp": pd.to_datetime("2024-01-01"),
                "sensors": {
                    f"{loc}-{d}": 3.25 for loc in LOCATIONS for d in DIRECTIONS
                }
            }
            for loc in LOCATIONS:
                self.machine_state[machine]["sensors"][f"{loc}-T"] = 42.0

        # Timestamp handling
        raw_time = raw_row.get("time stamp")
        if pd.isna(raw_time) or str(raw_time).strip() in ("", "nan", "None", "NULL"):
            # Advance last known timestamp by 1 day
            timestamp = self.machine_state[machine]["last_timestamp"] + pd.Timedelta(days=1)
            self.imputed_fields_count += 1
        else:
            timestamp = pd.to_datetime(raw_time)

        self.machine_state[machine]["last_timestamp"] = timestamp
        cleaned_row["time stamp"] = timestamp.strftime("%Y-%m-%d")

        # ----------------------------------------------------------------------
        # 2. Stateful Imputation of Vibration & Temperature Readings
        # ----------------------------------------------------------------------
        sensor_readings = {}
        for loc in LOCATIONS:
            # Directions V, H, A
            for d in DIRECTIONS:
                col_name = f"{loc}-{d}"
                val = raw_row.get(col_name)
                if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "None"):
                    # Use last known valid measurement from state
                    val = self.machine_state[machine]["sensors"][col_name]
                    self.imputed_fields_count += 1
                else:
                    val = float(val)
                    self.machine_state[machine]["sensors"][col_name] = val
                sensor_readings[col_name] = round(val, 2)
                cleaned_row[col_name] = round(val, 2)

            # Temperature T
            col_t = f"{loc}-T"
            val_t = raw_row.get(col_t)
            if pd.isna(val_t) or val_t is None or str(val_t).strip() in ("", "nan", "None"):
                val_t = self.machine_state[machine]["sensors"][col_t]
                self.imputed_fields_count += 1
            else:
                val_t = float(val_t)
                self.machine_state[machine]["sensors"][col_t] = val_t
            sensor_readings[col_t] = round(val_t, 2)
            cleaned_row[col_t] = round(val_t, 2)

        # ----------------------------------------------------------------------
        # 3. Real-Time Zone C Detection
        # ----------------------------------------------------------------------
        loc_max_vibs = {}
        for loc in LOCATIONS:
            v_val = sensor_readings[f"{loc}-V"]
            h_val = sensor_readings[f"{loc}-H"]
            a_val = sensor_readings[f"{loc}-A"]
            loc_max_vibs[loc] = max(v_val, h_val, a_val)

        dominant_location = max(loc_max_vibs, key=loc_max_vibs.get)
        max_machine_vib = loc_max_vibs[dominant_location]

        # Max temperature across all bearings
        max_machine_temp = max(sensor_readings[f"{loc}-T"] for loc in LOCATIONS)
        dominant_loc_temp = sensor_readings[f"{dominant_location}-T"]

        # Count locations entering Zone C
        locations_in_zone_c = [
            loc for loc, vib in loc_max_vibs.items() if vib >= self.zone_c_threshold
        ]
        num_locs_in_zone_c = len(locations_in_zone_c)

        in_zone_c = max_machine_vib >= self.zone_c_threshold

        cleaned_row["max_vibration"] = max_machine_vib
        cleaned_row["dominant_location"] = dominant_location
        cleaned_row["max_temperature"] = max_machine_temp
        cleaned_row["in_zone_c"] = in_zone_c

        # ----------------------------------------------------------------------
        # 4. Real-Time 3-Way Fault Diagnosis
        # ----------------------------------------------------------------------
        if not in_zone_c:
            cleaned_row["status"] = "NORMAL (Zone A/B)"
            cleaned_row["diagnosed_problem"] = "None"
        else:
            self.zone_c_alert_count += 1
            cleaned_row["status"] = "ALERT: ZONE C EXCEEDED"

            dom_rad_max = max(
                sensor_readings[f"{dominant_location}-V"],
                sensor_readings[f"{dominant_location}-H"]
            )
            dom_axial = sensor_readings[f"{dominant_location}-A"]
            axial_ratio = dom_axial / dom_rad_max if dom_rad_max > 0 else 1.0

            # Rule 1: High Frequency Vibration with High Temperature
            if max_machine_temp > self.temp_threshold or dominant_loc_temp > self.temp_threshold:
                diagnosis = "High Frequency Vibration with High Temperature"
            # Rule 2: Misalignment (Coordinated high across multiple bearings with strong axial)
            elif num_locs_in_zone_c >= 3 and axial_ratio >= 0.70:
                diagnosis = "Misalignment"
            # Rule 3: Imbalance (Dominant radial vibration, localized to rotor/pump, normal temp)
            else:
                diagnosis = "Imbalance"

            cleaned_row["diagnosed_problem"] = diagnosis
            self.fault_breakdown[diagnosis] += 1

        return cleaned_row


def stream_dataset(csv_path, max_rows=None, delay_seconds=0.0, verbose_zone_c_only=False):
    """
    Streams a dataset row-by-row (1 row = 1 streaming event).

    Parameters:
      csv_path: Path to dataset CSV
      max_rows: Optional limit on number of rows to stream (None = all rows)
      delay_seconds: Artificial delay between rows to simulate live telemetry
      verbose_zone_c_only: If True, prints only Zone C alerts; if False, prints all stream events
    """
    print(f"\n{'='*80}")
    print(f"STARTING STREAMING INGESTION: {csv_path}")
    print(f"Trigger Threshold: >= {ZONE_C_THRESHOLD_MM_S:.1f} mm/s RMS (Zone C) | Temp Limit: > {TEMP_HIGH_THRESHOLD_C:.1f} °C")
    print(f"{'='*80}")

    df = pd.read_csv(csv_path)
    total_to_stream = len(df) if max_rows is None else min(len(df), max_rows)

    processor = StreamingPumpProcessor()
    alerts = []

    print(f"Streaming {total_to_stream} rows (1 row per stream packet)...\n")

    for i in range(total_to_stream):
        raw_row = df.iloc[i].to_dict()
        event = processor.process_stream_event(raw_row, row_index=i)

        if event["in_zone_c"]:
            alerts.append(event)
            # Formatted live alert
            print(
                f"[STREAM #{i+1:04d}] >>> ZONE C ALERT <<< "
                f"Date: {event['time stamp']} | Machine: {event['Machine_number']} | "
                f"Bearing: {event['dominant_location']} ({event['max_vibration']:.2f} mm/s) | "
                f"Temp: {event['max_temperature']:.1f}°C -> "
                f"DIAGNOSIS: [ {event['diagnosed_problem'].upper()} ]"
            )
        else:
            if not verbose_zone_c_only and (i < 10 or (i + 1) % 500 == 0):
                print(
                    f"[STREAM #{i+1:04d}] Status: OK | "
                    f"Date: {event['time stamp']} | Machine: {event['Machine_number']} | "
                    f"Max Vib: {event['max_vibration']:.2f} mm/s | Temp: {event['max_temperature']:.1f}°C"
                )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    print("\n" + "-"*80)
    print("STREAMING INGESTION SUMMARY:")
    print(f"  * Total Rows Streamed: {processor.total_stream_count}")
    print(f"  * Missing Fields Imputed on-the-fly: {processor.imputed_fields_count}")
    print(f"  * Zone C Incidents Detected: {processor.zone_c_alert_count} ({processor.zone_c_alert_count/processor.total_stream_count:.2%})")
    print("  * Root Cause Breakdown in Stream:")
    for prob, count in processor.fault_breakdown.items():
        print(f"      - {prob}: {count} events")
    print("-"*80)

    return alerts, processor.fault_breakdown


if __name__ == "__main__":
    # Quick standalone test on the High Frequency + High Temperature dataset
    test_file = "pump_high_frequency_high_temperature_dataset_2024_2025.csv"
    if os.path.exists(test_file):
        stream_dataset(test_file, max_rows=1000)
