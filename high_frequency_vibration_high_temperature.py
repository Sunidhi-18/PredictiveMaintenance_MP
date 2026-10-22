"""
High Frequency Vibration with High Temperature Simulation
--------------------------------------------------------
This module models the third critical machinery fault condition:
High Frequency Vibration accompanied by High Bearing Temperature.

In industrial pumps and rotating machinery (as referenced in ISO 10816-7 and
ISO 13373 condition monitoring guidelines):
1. Rolling element bearing defects and mechanical rubbing generate high-frequency
   impacts, harmonics, and defect frequencies (e.g. BPFO, BPFI, orders > 3x up to 500+ Hz).
2. Friction, inadequate lubrication, and bearing degradation cause substantial
   temperature escalation (temperatures exceeding normal operating baseline of 35-50°C,
   reaching 70°C to 90°C+).

This script generates:
  1. pump_high_frequency_spectrum.csv (Frequency spectrum with high frequency components)
  2. pump_high_frequency_spectrum.png (12-panel spectral plots with defect markers)
  3. pump_high_frequency_high_temperature_dataset_2024_2025.csv (Daily telemetry with ~8% missing values)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# PART 1: Frequency Spectrum Simulation (High Frequency Vibration & Temperature)
# ==============================================================================
rng = np.random.default_rng(seed=42)

machine_number = "M1201"
rpm = 3000
running_frequency_hz = rpm / 60  # 50 Hz
measurement_timestamp = pd.Timestamp.now().isoformat(timespec="seconds")

# Sampling setup: sample rate 2048 Hz, 4 seconds -> 0.25 Hz resolution
sample_rate_hz = 2048
duration_seconds = 4
time = np.arange(0, duration_seconds, 1 / sample_rate_hz)

max_frequency_hz = 500
plot_max_frequency_hz = 500
noise_standard_deviation = 0.08  # mm/s

# Bearing defect frequencies (relative to running speed 1x = 50 Hz):
# BPFO (Outer Race): ~3.58x = 179 Hz
# BPFI (Inner Race): ~5.42x = 271 Hz
# High-frequency harmonics: 4x, 5x, 6x, 7x, 8x, 9x, 10x
# P2 bearing is experiencing severe defect with localized high friction & temperature
harmonic_rms = {
    "P1": {
        "H": {1: 1.8, 2: 0.6, 3: 0.4, 3.58: 1.2, 5.42: 1.5, 7.16: 1.1, 8: 0.8},
        "V": {1: 1.6, 2: 0.5, 3: 0.4, 3.58: 1.1, 5.42: 1.4, 7.16: 1.0, 8: 0.7},
        "A": {1: 1.2, 2: 0.4, 3: 0.3, 3.58: 0.9, 5.42: 1.0, 7.16: 0.8, 8: 0.5},
    },
    "P2": {  # Faulty bearing: High frequency vibration pushing overall RMS into Zone C (> 7.5 mm/s)
        "H": {1: 2.2, 2: 0.8, 3: 0.5, 3.58: 4.8, 4: 2.1, 5.42: 5.2, 6: 1.9, 7.16: 3.6, 8: 2.4, 10: 1.8},
        "V": {1: 2.0, 2: 0.7, 3: 0.5, 3.58: 4.5, 4: 1.9, 5.42: 4.9, 6: 1.7, 7.16: 3.4, 8: 2.2, 10: 1.6},
        "A": {1: 1.5, 2: 0.5, 3: 0.4, 3.58: 3.2, 4: 1.4, 5.42: 3.6, 6: 1.2, 7.16: 2.5, 8: 1.6, 10: 1.2},
    },
    "M1": {
        "H": {1: 1.5, 2: 0.4, 3: 0.3, 3.58: 0.8, 5.42: 0.9, 7.16: 0.6, 8: 0.4},
        "V": {1: 1.3, 2: 0.4, 3: 0.3, 3.58: 0.7, 5.42: 0.8, 7.16: 0.5, 8: 0.4},
        "A": {1: 1.0, 2: 0.3, 3: 0.2, 3.58: 0.5, 5.42: 0.6, 7.16: 0.4, 8: 0.3},
    },
    "M2": {
        "H": {1: 1.2, 2: 0.3, 3: 0.2, 3.58: 0.5, 5.42: 0.5, 7.16: 0.4, 8: 0.3},
        "V": {1: 1.1, 2: 0.3, 3: 0.2, 3.58: 0.4, 5.42: 0.5, 7.16: 0.3, 8: 0.2},
        "A": {1: 0.9, 2: 0.2, 3: 0.2, 3.58: 0.3, 5.42: 0.4, 7.16: 0.2, 8: 0.2},
    },
}

# Bearing temperatures (P2 exhibits severe overheating due to defect friction)
bearing_temperatures_c = {
    "P1": 46.5,
    "P2": 84.2,  # Abnormally HIGH TEMPERATURE (> 65°C / 75°C threshold)
    "M1": 48.0,
    "M2": 44.1,
}

def create_velocity_signal(order_amplitudes):
    """Create synthetic velocity signal with high-frequency harmonics and noise."""
    signal = np.zeros_like(time)
    for order, rms_amplitude in order_amplitudes.items():
        freq_hz = order * running_frequency_hz
        peak = rms_amplitude * np.sqrt(2)
        phase = rng.uniform(0, 2 * np.pi)
        signal += peak * np.sin(2 * np.pi * freq_hz * time + phase)
    
    # Add high-frequency random impacts/carpet noise
    signal += rng.normal(loc=0.0, scale=noise_standard_deviation, size=len(time))
    return signal

def calculate_rms_spectrum(signal):
    """Calculate single-sided RMS amplitude spectrum."""
    fft_values = np.fft.rfft(signal)
    frequencies_hz = np.fft.rfftfreq(len(signal), d=1 / sample_rate_hz)
    peak_amplitude = 2 * np.abs(fft_values) / len(signal)
    peak_amplitude[0] /= 2
    if len(signal) % 2 == 0:
        peak_amplitude[-1] /= 2
    rms_amplitude = peak_amplitude / np.sqrt(2)
    return frequencies_hz, rms_amplitude

records = []
spectra_for_plot = {}

for location, directions in harmonic_rms.items():
    loc_temp = bearing_temperatures_c[location]
    for direction, order_amps in directions.items():
        sig = create_velocity_signal(order_amps)
        freqs, spec_rms = calculate_rms_spectrum(sig)
        overall_rms = np.sqrt(np.mean(sig ** 2))
        spectra_for_plot[(location, direction)] = (freqs, spec_rms)

        for freq, amp in zip(freqs, spec_rms):
            if freq <= max_frequency_hz:
                records.append({
                    "Machine_number": machine_number,
                    "Timestamp": measurement_timestamp,
                    "RPM": rpm,
                    "Running_frequency_Hz": running_frequency_hz,
                    "Location": location,
                    "Direction": direction,
                    "Frequency_Hz": round(float(freq), 3),
                    "Frequency_Order": round(float(freq / running_frequency_hz), 3),
                    "Velocity_RMS_mm_s": round(float(amp), 5),
                    "Overall_RMS_mm_s": round(float(overall_rms), 3),
                    "Bearing_Temperature_C": loc_temp,
                    "Condition": "High_Frequency_Vibration_High_Temperature" if location == "P2" else "Normal"
                })

df_spectrum = pd.DataFrame(records)
spectrum_csv = "pump_high_frequency_spectrum.csv"
df_spectrum.to_csv(spectrum_csv, index=False)
print(f"[1/3] Spectrum dataset saved to: {spectrum_csv}")

# Plot 12-panel spectra
locations = ["P1", "P2", "M1", "M2"]
directions = ["H", "V", "A"]

fig, axes = plt.subplots(nrows=len(locations), ncols=len(directions), figsize=(16, 12), sharex=True)

for r_idx, loc in enumerate(locations):
    temp = bearing_temperatures_c[loc]
    for c_idx, dir_name in enumerate(directions):
        ax = axes[r_idx, c_idx]
        freqs, spec_rms = spectra_for_plot[(loc, dir_name)]
        vis = freqs <= plot_max_frequency_hz
        
        ax.plot(freqs[vis], spec_rms[vis], linewidth=0.9, color="#d95f02" if loc == "P2" else "#1f77b4")
        
        # Mark high-frequency defect orders (BPFO 3.58x, BPFI 5.42x, 2xBPFO 7.16x, 8x)
        defect_orders = [3.58, 5.42, 7.16, 8.0]
        for do in defect_orders:
            dfreq = do * running_frequency_hz
            ax.axvline(dfreq, color="crimson", linestyle=":", alpha=0.6)
            if r_idx == 1:
                ax.text(dfreq, 0.88, f"{do:.1f}x", transform=ax.get_xaxis_transform(), color="crimson", fontsize=7)

        title = f"{loc} - {dir_name} (Temp: {temp}°C)"
        if loc == "P2":
            title += " [FAULT: High Freq + High Temp]"
        ax.set_title(title, fontsize=9, fontweight="bold" if loc == "P2" else "normal")
        ax.set_ylabel("Velocity (mm/s RMS)", fontsize=8)
        ax.set_xlim(0, plot_max_frequency_hz)
        ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("Frequency (Hz)", fontsize=10)

fig.suptitle(
    f"High Frequency Vibration with High Temperature Spectrum Analysis\n"
    f"Machine {machine_number} | 3000 RPM (1x = 50 Hz) | Severe Defect at P2 (T={bearing_temperatures_c['P2']}°C)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plot_file = "pump_high_frequency_spectrum.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()
print(f"[2/3] Spectral plot saved to: {plot_file}")

# ==============================================================================
# PART 2: Daily Telemetry Dataset Simulation (High Frequency + High Temperature)
# ==============================================================================
machine_numbers = [f"M{num}" for num in range(1201, 1206)]
start_date = "2024-01-01"
end_date = "2025-12-31"
dates = pd.date_range(start=start_date, end=end_date, freq="D")

locations_daily = ["P1", "P2", "P3", "P4"]
directions_daily = ["V", "H", "A"]

# High frequency + high temperature fault events:
# Distinct events where specific bearings experience severe degradation:
# Event 1: M1202 on P2 from 2024-08-10 to 2024-08-14
# Event 2: M1204 on P3 from 2025-03-20 to 2025-03-24
event_configs = {
    ("M1202", "P2"): pd.date_range("2024-08-10", "2024-08-14"),
    ("M1204", "P3"): pd.date_range("2025-03-20", "2025-03-24"),
    ("M1201", "P1"): pd.date_range("2025-10-05", "2025-10-08"),
}

records_daily = []

for m_idx, machine in enumerate(machine_numbers):
    m_temp_base = rng.uniform(-1.0, 1.0)
    for d_idx, dt in enumerate(dates):
        row = {
            "Machine_number": machine,
            "time stamp": dt,
            "_high_fault_day": False
        }

        seasonal_temp = 3.0 * np.sin(2 * np.pi * d_idx / 365.25)

        for loc_idx, loc in enumerate(locations_daily):
            is_fault = (machine, loc) in event_configs and dt in event_configs[(machine, loc)]

            if is_fault:
                row["_high_fault_day"] = True
                # Vibration enters Zone C (7.6 - 9.8 mm/s) localized to this bearing
                v_center = rng.uniform(7.8, 9.6)
                # Temperature escalates into high zone: 72°C - 88°C
                temp = rng.uniform(74.0, 88.0)
            else:
                # Normal baseline vibration (Zone A/B: 3.1 - 4.5 mm/s)
                v_center = rng.normal(loc=3.35, scale=0.25)
                # Normal baseline temperature (38°C - 50°C)
                temp = (
                    42.0
                    + m_temp_base
                    + loc_idx * 0.6
                    + seasonal_temp
                    + rng.normal(0, 1.0)
                )
                temp = np.clip(temp, 30.0, 60.0)

            # Directional variations: high frequency impacts appear across V, H, with moderate A
            offsets = rng.uniform(-0.35, 0.35, size=3)
            vibs = np.clip(v_center + offsets, 2.5, 12.0)

            row[f"{loc}-V"] = round(float(vibs[0]), 2)
            row[f"{loc}-H"] = round(float(vibs[1]), 2)
            row[f"{loc}-A"] = round(float(vibs[2]), 2)
            row[f"{loc}-T"] = round(float(temp), 2)

        records_daily.append(row)

df_telemetry = pd.DataFrame(records_daily)

# Arrange columns exactly matching the standard telemetry schema
ordered_columns = ["Machine_number", "time stamp"]
for loc in locations_daily:
    ordered_columns.extend([f"{loc}-V", f"{loc}-H", f"{loc}-A", f"{loc}-T"])

# Inject ~8% missing values, protecting fault event rows to ensure fault signature preservation
vibration_columns = [f"{loc}-{d}" for loc in locations_daily for d in directions_daily]
temperature_columns = [f"{loc}-T" for loc in locations_daily]

protected = pd.DataFrame(False, index=df_telemetry.index, columns=df_telemetry.columns)
fault_rows = df_telemetry["_high_fault_day"]
protected.loc[fault_rows, vibration_columns + temperature_columns] = True

df_clean_subset = df_telemetry[ordered_columns].copy()
eligible = np.argwhere(~protected[ordered_columns].to_numpy())
missing_target = round(df_clean_subset.size * 0.08)

chosen = rng.choice(len(eligible), size=missing_target, replace=False)
for pos in eligible[chosen]:
    r, c = pos
    df_clean_subset.iat[r, c] = np.nan

telemetry_csv = "pump_high_frequency_high_temperature_dataset_2024_2025.csv"
df_clean_subset.to_csv(telemetry_csv, index=False)

missing_pct = df_clean_subset.isna().sum().sum() / df_clean_subset.size * 100
print(f"[3/3] Daily telemetry dataset saved to: {telemetry_csv}")
print(f"      Rows: {len(df_clean_subset)}, Columns: {len(df_clean_subset.columns)}")
print(f"      Missing Values: {df_clean_subset.isna().sum().sum()} ({missing_pct:.2f}%)")
print(f"      Fault Events Simulated: {len(event_configs)} multi-day degradation events")
