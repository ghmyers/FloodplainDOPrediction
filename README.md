# FloodplainDOPrediction

A deep learning framework for predicting floodplain dissolved oxygen using time-series data and transfer learning. 

---

## 🌍 Motivation

Floodplain ecosystems are highly sensitive to oxygen fluctuations, with hypoxic events changing dominant biogeochemical processes and threatening aquatic life. This project uses **LSTM-based recurrent neural networks** and **transfer learning** to predict dissolved oxygen (DO) levels in floodplain wetland sites.

We demonstrate that cross-site transfer learning significantly improves model performance in **data-scarce floodplain settings**.

---

## 🧠 Methods

- **Model Architecture**: LSTM-based time series forecasting  
- **Transfer Learning Strategy**: Pre-train on one 480 river sites across the CONUS, fine-tune on seven floodplain sites in the Lake Champlain Basin of Vermont.  
- **Model Comparison**: Compared a river LSTM (trained only on river data), a floodplain LSTM (trained only on floodplain data), and a transfer learning LSTM
- **Input Features**: 
  - Daily streamflow and water temperature
  - Meteorological variables
  - Static Catchment attributes
- **Target Variable**: DO

## 📂 Project Structure

### **🔹 Directory Breakdown**
📌 **`data/`** → Stores all datasets for the project.
- **`raw/`** → Unprocessed data as received from the source.
- **`processed/`** → Data that has been cleaned and preprocessed.
- **`metadata/`** → Configuration files, data dictionaries, or metadata about datasets.

📌 **`notebooks/`** → Jupyter notebooks for analysis, data exploration, and experimentation.

📌 **`scripts/`** → Python scripts for automation, data preprocessing, and model training.

📌 **`src/`** → Source code for the project.
- **`utils/`** → Helper functions such as logging, preprocessing utilities, and feature engineering.

📌 **`models/`** → Saved machine learning models and model checkpoints.

📌 **`outputs/`** → Stores generated reports, plots, visualizations, and final results.

📌 **`logs/`** → Logging files for tracking the execution of scripts.

📌 **`config/`** → Configuration files (e.g., `.yaml`, `.json`) for model and script settings.

📌 **`tests/`** → Unit tests to validate scripts and model performance.

---

## 🧠 Author

**George Harrison Myers**  
PhD Student | Machine Learning Engineer | Environmental Data Scientist
[LinkedIn](https://www.linkedin.com/in/harrison-myers-eit-b37156181/) • [Email](mailto:ghmyers96@gmail.com) • [GitHub](https://github.com/finnmyers96)

---
