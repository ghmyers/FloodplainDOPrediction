# 🌊 From Rivers to Floodplains: Leveraging Transfer Learning to Predict Floodplain Dissolved Oxygen

This repository contains the code and workflows used in the study:  
**"From Rivers to Floodplains: Leveraging Transfer Learning to Predict Floodplain Dissolved Oxygen"**  
*(Myers et al., 2025, preprint link below).*

---

## 📍 Project Overview

Floodplain dissolved oxygen (DO) dynamics are critical for nutrient cycling and ecosystem health but are difficult to predict due to limited monitoring data. This project applied a **transfer learning approach using deep learning models** trained on well-monitored river data and fine-tuned for floodplain conditions. We compared models trained only on river data, only on floodplain data, and a transfer learning model trained on both, finding that transfer learning improved predictions—especially during low DO events. To better understand model behavior, we used **explainable AI techniques** to interpret the predictions and highlight shared process knowledge between rivers and floodplains.

![Conceptual Model of Floodplain DO](figures/floodplain_DO_conceptual_model)
*Simplified conceptual model of floodplain DO dynamics as mediated by hydrologically connected riverine water quality and quantity (Created in BioRender. Myers, H. (2025) https://BioRender.com/y49l602)*

We compared three models:  
1️⃣ **River LSTM Model** – Trained solely on well-monitored river datasets.  
2️⃣ **Floodplain LSTM Model** – Trained only on scarce floodplain data.  
3️⃣ **TL LSTM Model** – Pre-trained on river data, then fine-tuned on floodplain data.  

Our findings highlight that **transfer learning significantly enhances DO prediction performance**, particularly during **low-oxygen periods**, which are critical for understanding biogeochemical processes and ecosystem health.

---

## 🧠 Methods

- **Deep Learning Model**: Long Short-Term Memory (LSTM) network  
- **Transfer Learning Approach**: Pre-train on **480 USGS river gages**, fine-tune on **7 floodplain sites**  
- **Feature Selection**: Dynamic hydrometeorological inputs + static catchment attributes  
- **Explainable AI**: SHAP values used to interpret model predictions  
- **Data Sources**: USGS, DayMET, NHD, and in-situ floodplain sensor data
- 

![Conceptual Model of Floodplain DO](figures/model_training_workflow.png)
*Conceptual diagram of methodological approach. a) river training dataset; b) floodplain dataset; c) feature selection of model inputs; d) river LSTM model; e) floodplain LSTM model; f) TL LSTM model. (Created in BioRender. Myers, H. (2025) https://BioRender.com/b73n623)*
📖 *Full methodological details are in the manuscript (linked below).*

---

## ⚙️ Tech Stack

- **Python** (TensorFlow, pandas, NumPy, scikit-learn, SHAP)  
- **Data Processing**: pandas, pyNHD 
---

## 🗃️ Data & Figures

This study used a combination of **publicly available datasets** and **in-situ floodplain DO observations**.  
- **River Training Data**: 480 USGS river gages (1980–2022)  
- **Floodplain Observations**: 7 floodplain sites in Vermont’s Otter Creek Basin (2019–2023)  
- **Meteorological Data**: Extracted from **DayMET**  
- **Catchment Attributes**: Processed via **PyNHD**  

📝 **Raw data is not included in this repository but can be found here (https://zenodo.org/records/14553375).**  
📌 **Preprocessing scripts are provided** to enable reproduction using open-access data.

---

## 📂 Project Structure

### **🔹 Directory Breakdown**

📌 **`notebooks/`** → Jupyter notebooks for analysis, data exploration, and experimentation.

📌 **`scripts/`** → Python scripts for feature selection, data cleaning, etc. 

📌 **`src/`** → Source code for the project.
- **`utils/`** → Helper functions such as model training/testing functions, data preprocessing, etc.

📌 **`models/`** → Saved machine learning models and model checkpoints.

📌 **`config/`** → Configuration files (e.g., `.yaml`, `.json`) for model and script settings.

---

## 📚 Citation
Myers, G. H., et al. (2025). From Rivers to Floodplains: Leveraging Transfer Learning to Predict Floodplain Dissolved Oxygen.
📄 Preprint Available Here (https://essopenarchive.org/users/879154/articles/1258431-from-rivers-to-floodplains-leveraging-transfer-learning-to-predict-floodplain-dissolved-oxygen)

---

## 🧠 Author

**George Harrison Myers**  
PhD Student | Machine Learning Engineer | Environmental Data Scientist
[LinkedIn](https://www.linkedin.com/in/harrison-myers-eit-b37156181/) • [Email](mailto:ghmyers96@gmail.com) • [GitHub](https://github.com/ghmyers)

---
