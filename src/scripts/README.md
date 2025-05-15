# Battery Voltage Forecasting: Model Evaluation Results

This document summarizes the performance of different time series models used for forecasting battery voltage (`Voltage_measured`) in the B0005 dataset. The initial benchmark uses a **Naive** forecasting model. Future results from more advanced models (e.g., ARIMA, ETS, Prophet, LSTM) will be added for comparison.

---

## 📊 Dataset
- **Dataset**: B0005 (Voltage measurements)
- **Frequency**: 1 minute
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

## 📉 Baseline Model: Naive Forecast

| Model | MAE     | MSE     | RMSE    | SMAPE   | MASE   | Bias    | Time Elapsed (s) |
|-------|---------|---------|---------|---------|--------|---------|------------------|
| Naive | 0.20178 | 0.06332 | 0.25164 | 5.90646 | 0.91556 | 2.78616 | 0.209            |

---

## 🚧 To Be Added
- [ ] **ARIMA**
- [ ] **Exponential Smoothing (ETS)**
- [ ] **Prophet**
- [ ] **LSTM**
- [ ] **XGBoost Regressor**

---

## 📁 Directory Structure
.
├── data/
│ ├── B0005_train.parquet
│ ├── B0005_val.parquet
│ └── B0005_test.parquet
├── src/
│ ├── models/
│ ├── scripts/
│ │ └── README.md ← (you are here)
│ └── utils/
├── requirements.txt
└── README.md

