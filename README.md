# Predictive Maintenance & Pump Vibration Zone C Diagnostics

Industrial pump condition monitoring system conforming to **BS ISO 10816-7** and **Europump Guidelines (2013)**. The system performs Apache Spark data cleaning on synthetic vibration telemetry and identifies root causes when machine readings cross into **Zone C**.

---

## 🚀 Key Features

1. **3-Way Fault Diagnosis Engine**:
   - **Imbalance**: $1\times$ running speed dominant peak, high radial vibration ($V, H$), normal temperature ($\le 65^\circ\text{C}$).
   - **Misalignment**: Multi-bearing coordinated vibration with strong axial component ($A/\max(V,H) \ge 0.70$), normal temperature.
   - **High Frequency Vibration with High Temperature**: High-frequency harmonics ($> 3\times$ up to $500\text{ Hz}$, bearing defect frequencies $\text{BPFO}/\text{BPFI}$) and abnormal bearing overheating ($> 65^\circ\text{C}$, reaching $74^\circ\text{C}-88^\circ\text{C}$).

2. **Standards-Based Zone C Thresholds (ISO 10816-7)**:
   - **Vibration Velocity Threshold**: $\ge 7.0\text{ mm/s}$ RMS (Zone C alarm trigger).
   - **Bearing Temperature Threshold**: $> 65.0^\circ\text{C}$ (thermal anomaly limit).

3. **Apache Spark (PySpark) Data Cleaning**:
   - Reconstructs missing `Machine_number` and `time stamp` deterministically.
   - Imputes missing vibration and temperature readings on time-series telemetry using Spark Window forward/backward fill.
   - Guaranteed $0$ missing values ($100\%$ clean).

4. **Real-Time Streaming Engine**:
   - Converts static datasets into continuous live telemetry where **1 row acts as 1 live stream packet**.
   - Performs stateful streaming imputation on-the-fly.
   - Emits live alerts and classifies incoming events into one of the 3 problems in real time.

---

## 📁 Repository Structure

| File | Description |
|---|---|
| `high_frequency_vibration_high_temperature.py` | Generates 3rd fault telemetry dataset, spectrum, and plots |
| `spark_data_cleaning.py` | PySpark pipeline to clean missing values across all datasets |
| `zone_c_fault_diagnostics.py` | PySpark batch diagnostic engine for ISO 10816-7 Zone C evaluation |
| `streaming_zone_c_diagnostics.py` | Core real-time streaming processor (1 row = 1 stream) |
| `verify_streaming.py` | Complete streaming verification script across all 3,655 rows |
| `verify_all.py` | End-to-end verification script for simulation, cleaning, and batch diagnostics |
| `Zone_C_Detection_and_Fault_Diagnostics.ipynb` | Interactive Jupyter notebook with diagnostics and charts |
| `Guidelines 0n Pump Vibration First edition Final July 2013.pdf` | Europump pump vibration standards guide |
| `ISO-10816-7-2009.pdf` | ISO 10816-7 mechanical vibration standard |

---

## 🛠️ How to Run

### 1. Verify Streaming Pipeline (1 Row = 1 Stream)
```powershell
python verify_streaming.py
```

### 2. Verify Batch Cleaning & Diagnostics
```powershell
python verify_all.py
```

### 3. Run PySpark Missing Value Cleaning
```powershell
python spark_data_cleaning.py
```

---

## 📊 Summary of Fault Classification

$$\textbf{Vibration} \ge 7.0\text{ mm/s} \implies \textbf{Zone C Warning Triggered}$$

- **High Freq + High Temp**: $\text{Vibration} \ge 7.0\text{ mm/s}$ **AND** $\text{Temperature} > 65.0^\circ\text{C}$
- **Misalignment**: $\text{Vibration} \ge 7.0\text{ mm/s}$, $\text{Temperature} \le 65.0^\circ\text{C}$, $\ge 3$ bearings elevated with high axial ratio ($\ge 0.70$)
- **Imbalance**: $\text{Vibration} \ge 7.0\text{ mm/s}$, $\text{Temperature} \le 65.0^\circ\text{C}$, radial dominant ($H/V$) with lower axial motion
