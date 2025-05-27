# Battery Voltage Forecasting: Model Evaluation Results

This document summarizes the performance of different time series models used for forecasting battery voltage (`Voltage_measured`) in the B0005 dataset. The initial benchmark uses a **Naive** forecasting model. Future results from more advanced models (e.g., ARIMA, ETS, Prophet, LSTM) will be added for comparison.

---

## 📊 Dataset
- **Dataset**: B0005 (Voltage measurements)
- **Frequency**: 15 seconds
- **Train/Validation/Test Split**: Preprocessed and stored as Parquet files.

---

## ✅ Evaluation Metrics
The following metrics are used for model evaluation:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error (seasonality = 48)
- **Bias**: Forecast bias

---

## 📉 Model Performance


| Model         | MAE     | MSE     | RMSE    | SMAPE   | MASE    | Bias    | Time Elapsed (s) |      |
|---------------|---------|---------|---------|---------|---------|---------|------------------|------|
| Naive         | 0.20178 | 0.06332 | 0.25164 | 5.90646 | 0.91556 | 2.78616 | 0.209            |      |
| SeasonalNaive | 0.32101 | 0.13971 | 0.37378 | 9.40310 | 1.45656 | 0.35230 | 0.199            |      |
|WindowAverage  | 0.18831 | 0.05874 | 0.24236 | 5.52418 | 0.85446 | 0.97464 | 0.194            |      |
| HoltWinters   | 0.18617 | 0.05830 | 0.24146 | 5.46085 | 0.84473 | 0.15466 | 9.107            |      |
| AutoETS       | 0.18617 | 0.05830 | 0.24146 | 5.46085 | 0.84473 | 0.15466 | 8.930            |      |
| ARIMA         | 0.22765 | 0.07834 | 0.27990 | 6.70182 | 1.03296 | 3.70380 | 113.202          |      |
| Theta         | 0.20265 | 0.06538 | 0.25569 | 5.94939 | 0.91951 | 2.52165 | 17.935           |      |
| TBATS         | 0.40598 | 0.21904 | 0.46801 | 12.35833| 1.84209 | 9.87590 | 13.419           |      |
| MSTL          | 0.20345 | 0.06649 | 0.25786 | 5.97373 | 0.92312 | 2.65154 | 2.807            |      |

Based on this sample test, the best performing models are (HoltWinters, and **ETS**), **ARIMA**, and **TBATS**. Lets build that for all models using **AutoETS** and **TBATS**. ARIMA gives similar performance to TBATS, but TBATS is faster.


✅ Forecast plots saved:
- Static PNG: `imgs/.png`
- Interactive HTML: `imgs/.html`

---

## 🚧 To Be Added
- [x] **ARIMA**
- [x] **Exponential Smoothing (ETS)** (AutoETS, HoltWinters)
- [ ] **Prophet**
- [ ] **LSTM**
- [ ] **XGBoost Regressor**

---

## 📁 Directory Structure
```


project-root/
├── README.md
├── compose.yml
├── Dockerfile
├── requirements.txt
├── data/
│   ├── B0005_discharge.csv
│   ├── B0005_discharge_adjusted.csv
│   ├── B0005_test.parquet
│   ├── B0005_train.parquet
│   ├── B0005_val.parquet
│   └── combined_discharge.csv
├── imgs/
│   ├── naive_forecast.html
│   └── seasonalnaive_forecast.html
├── Research/
│   ├── test.ipynb
│   ├── test_g.ipynb
│   ├── test_s.ipynb
│   ├── CAPSTONE PROJECT - TFT for multiple series/
│   ├── N-BEATS/
│   └── TFT/
└── src/
    ├── __init__.py
    ├── __pycache__/
    ├── config/
    ├── decomposition/
    ├── models/
    ├── scripts/
    │   ├── README.md ← (you are here)
    │   ├── ch04.py
    │   ├── ch0402.py
    │   ├── ch0402_Naive.py
    │   ├── ch0402_SeasonalNaive.py
    │   ├── ch04_1.py
    │   ├── p02.py
    │   └── p03.py
    └── utils/

```
