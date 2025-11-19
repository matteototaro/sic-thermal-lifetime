# Lifetime Modeling of SiC Power Modules in Automotive Traction Inverters (Python Implementation)

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AQG 324](https://img.shields.io/badge/Standard-AQG%20324-green.svg)](https://www.ecpe.org)

---

## 📋 Overview

This repository contains the comprehensive **Python implementation** for predicting thermal fatigue lifetime in **Silicon Carbide (SiC) Power MOSFETs** used in high-performance automotive traction inverters.

The work successfully combines:
- ⚡ **Software modeling** (Python tool with MATLAB thesis model)
- 🔬 **Experimental characterization** methods
- 📊 **Industry-standard compliance** (ECPE AQG 324)

![Thermal Lifetime Analysis](example_output.png)

---

## 🎯 Motivation and Context

The automotive industry's transition to **high-performance Electric Vehicles (EVs)** demands:
- Smaller, lighter, more powerful components
- **Silicon Carbide (SiC)** technology for superior performance
- **Long-term reliability** under extreme operating conditions

---

## 🚀 What This Tool Does

### Convert Driving Cycles → Lifetime Prediction

```
Electric Motor Profile (Speed, Torque, Time)
              ↓
    [Electro-Thermal Model]
              ↓
┌─────────────────────────────────────┐
│  Coupled Iterative Solution Loop:   │
│                                     │
│  T_j(i-1) → P_loss(i) → ΔT → T_j(i) │
│      ↑                         ↓    │
│      └─────────────────────────┘    │
│                                     │
│  Converges when thermal stability   │
│  is reached at each time step       │
└─────────────────────────────────────┘
              ↓
    Junction Temperature (T_j)
              ↓
    [Rainflow Cycle Counting]
              ↓
     Thermal Stress Cycles
              ↓
    [Bayerer Model - CIPS08]
    ΔT_j, T_j_max, t_on → N_f
              ↓
   Cumulative Damage (Miner's Rule)
              ↓
 ✓ Lifetime Consumption: X.XX%
 ✓ Extrapolated Life: XXXX hours
 ✓ AQG 324: PASS/FAIL
```

### Three Mission Profiles

The tool analyzes three standardized automotive scenarios with **progressive intensity**:

| Scenario | Speed Range | Torque Range | Thermal Stress | Use Case |
|----------|------------|--------------|----------------|----------|
| **Urban** | 0-8000 RPM | 10-50 Nm | Low | City driving, stop-and-go |
| **Highway** | 5000-14000 RPM | 20-320 Nm | Medium | Highway cruise, overtaking |
| **Performance** | 0-15000 RPM | 20-450 Nm | High | Track driving, max acceleration |

---

## 📦 Installation

### Prerequisites

```bash
Python >= 3.7
pip >= 20.0
```

### Quick Install

```bash
# Clone repository
git clone https://github.com/matteototaro/sic-thermal-lifetime.git
cd sic-thermal-lifetime

# Install dependencies
pip install -r requirements.txt

# Run the tool
python thermal_lifetime_prediction.py
```

### Dependencies

```
numpy>=1.19.0
pandas>=1.0.0
matplotlib>=3.2.0
scipy>=1.5.0
tqdm>=4.50.0  # Optional, for progress bars
```

---

## 📊 Usage

### Basic Execution

Simply run the script - it will automatically analyze all three scenarios:

```bash
python thermal_lifetime_prediction.py
```

The script will:
1. Load pre-generated mission profiles from CSV files (in `profiles/` folder)
2. Perform coupled electro-thermal simulation
3. Extract thermal cycles using rainflow counting
4. Calculate cumulative damage and lifetime
5. Generate comprehensive visualization

### Output

The tool generates:

#### 1. Console Output
```
================================================================================
  LIFETIME CONSUMPTION: 0.012500%
================================================================================
  Extrapolated: 8000 hours (0.9 years)
  AQG 324: 0.13 / 1000 PC cycles
  ✓ PASS - Margin: 999.87 cycles (99.99%)
```

#### 2. Comprehensive Visualization
A publication-quality comparison plot (`example_output.png`) showing:
- **Row 1:** Motor Speed profiles (all 3 scenarios side-by-side)
- **Row 2:** Motor Torque profiles (all 3 scenarios)
- **Row 3:** Junction Temperature evolution (all 3 scenarios)
- **Row 4:** Comparative summary table with all key metrics

#### 3. Results Dictionary

```python
results = run_lifetime_test_all_scenarios()

# Access individual scenario data
profile, T_j, P_loss, cycles, damage_df, total_damage = results['urban']
profile, T_j, P_loss, cycles, damage_df, total_damage = results['highway']
profile, T_j, P_loss, cycles, damage_df, total_damage = results['performance']
```

---

## 🔬 Technical Methodology

### 1. Electro-Thermal Coupling

The tool implements proper feedback between electrical and thermal domains through an iterative process at each time step:

```
T_j(i-1) → P_loss(i) → Thermal Network → T_j(i)
```

**Power Loss Components:**
- Switching losses: `E_on(T_j) × f_sw`
- Conduction losses: `I_rms² × R_DS_on(T_j)`
- Diode losses: `V_F × I_avg`

All temperature-dependent parameters are updated at each iteration based on the previous junction temperature.

### 2. Thermal Network

4-layer RC network using **Explicit Euler integration**:

```
C_th × dT/dt = Q_in - Q_out
T(t+Δt) = T(t) + Δt × dT/dt
```

**Stability condition:** `Δt < 2 × min(R_th × C_th)`

### 3. Rainflow Cycle Counting

Implements **ASTM E1049-85** standard:
- Extracts fatigue-relevant thermal cycles
- Tracks temperature swing (ΔT), mean temperature, and dwell time
- Converts continuous temperature history into discrete damage events

### 4. Lifetime Prediction - Bayerer Model (CIPS08)

**Power Cycling Lifetime Model** (Bayerer et al., 2008):

```
N_f = K × (ΔT_j)^β₁ × exp(β₂/T_j_max) × t_on^β₃
```

Where:
- **β₁ = -3.483** (Coffin-Manson: thermal strain dependency)
- **β₂ = 1917 K** (Arrhenius: temperature activation energy)
- **β₃ = -0.438** (Time-dependent creep/diffusion effects)
- **ΔT_j**: Junction temperature swing [K]
- **T_j_max**: Maximum junction temperature [K]
- **t_on**: Heating time / power-on duration [s]
- **K**: Technology-dependent constant calibrated from test data

**Note:** The original Bayerer (CIPS08) model includes a fourth parameter (β₄) for current per bond wire. However, for a fixed module technology and specific application, this parameter becomes a constant absorbed into K. Therefore, this implementation uses the simplified three-parameter form commonly applied in automotive qualification standards.

**Acceleration Factor (without K, comparing field to test conditions):**

The acceleration factor relates field conditions to test conditions:

```
AF = (ΔT_field / ΔT_test)^β₁ × exp[β₂ × (1/T_max_field - 1/T_max_test)] × (t_on_field / t_on_test)^β₃

N_f_field = N_f_test × AF
```

This allows prediction of lifetime under actual operating conditions based on standardized qualification test results.

**Miner's Rule (Cumulative Damage):**

```
D_total = Σ(n_i / N_f_i)
```

Failure predicted when `D_total ≥ 1.0` (100% life consumed)


## ⚠️ NOTE ON CONFIDENTIALITY

This work was initially developed on MALTAB in collaboration with **Ferrari S.p.A.** for my Master's Thesis.

Due to intellectual property (IP) considerations:
- ✅ **Full thesis documentation** is publicly available
- ✅ **Python demonstration tool** with generic parameters is provided
- ❌ **Production MATLAB model** with Ferrari-specific data remains confidential

The Python tool uses **publicly available Infineon datasheet parameters** to demonstrate the complete methodology while respecting confidentiality requirements.

---

## 📄 Accessing the Thesis

📖 **Full Thesis PDF:** [Totaro_Matteo_MastersThesis.pdf]([text](https://github.com/matteototaro/Fatigue-Analysis-SiC-Power-Module/blob/main/Totaro_Matteo_MastersThesis.pdf))

The thesis includes:
- Comprehensive literature review on SiC power modules
- Detailed mathematical derivation of lifetime models
- Experimental test setup and procedures
- Results validation and sensitivity analysis
- Future work and recommendations


## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{totaro2022sic,
  author  = {Totaro, Matteo},
  title   = {Lifetime Modeling of SiC Power Modules in Automotive Traction Inverters},
  school  = {ALMA MATER STUDIORUM - Università di Bologna},
  year    = {2022},
  type    = {Master's Thesis},
  note    = {In collaboration with Ferrari S.p.A.}
}
```
---

## 📖 References

1. **ECPE AQG 324** - Qualification of Automotive Electronic Components
2. **Infineon FS03MR12A6MA1B** Datasheet - 1200V 310A SiC Power Module
3. **ASTM E1049-85** - Standard Practices for Cycle Counting in Fatigue Analysis
4. **Bayerer, R., Herrmann, T., Licht, T., Lutz, J., & Feller, M. (2008)** - "Model for Power Cycling lifetime of IGBT Modules - various factors influencing lifetime", *5th International Conference on Integrated Power Electronics Systems (CIPS)*, Nuremberg, Germany, pp. 1-6
5. **Miner (1945)** - Cumulative Damage in Fatigue, *Journal of Applied Mechanics*
6. **IEC 60749-34** - Power Cycling Test to Failure for Power Semiconductor Devices
7. **Held, M., et al. (1997)** - "Fast power cycling test of IGBT modules in traction application", *LESIT Project*

---

## 🔧 Contact

**Matteo Totaro**  
Email: tmatteos@gmail.com  
Website: [mtotaro.com](https://mtotaro.com)  
LinkedIn: [linkedin.com/in/m-totaro](https://linkedin.com/in/m-totaro)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.