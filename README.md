# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
# 🚗 AlphaCare Insurance Analytics Challenge  
### *Risk Analysis • Predictive Modeling • Pricing Optimization*

This repository contains my full work submission for the **AlphaCare Insurance Solutions (ACIS)** analytics challenge. The goal of the project is to analyse **historical car-insurance claim data** (Feb 2014 – Aug 2015) and deliver insights, statistical evidence, and machine-learning models that help ACIS identify **low-risk customers**, optimise **premium pricing**, and improve **marketing strategy**.

---

## 📌 Project Objectives

- Understand risk drivers through EDA  
- Evaluate statistical differences between customer groups  
- Build predictive models for claim severity and premium pricing  
- Implement DVC for reproducible data versioning  
- Maintain proper Git/GitHub workflow  
- Produce a final report (max 10 pages) summarising insights & models  

---

## 📁 Repository Structure

project-root/
│
├── data/ # Raw and processed data (DVC-tracked)
├── notebooks/ # EDA, hypothesis tests, model development
├── src/ # Scripts for preprocessing, modeling, utils
├── reports/ # Interim and final reports
├── dvc.yaml # DVC pipeline definition
├── requirements.txt # Python dependencies
└── README.md


---

# ✅ Task 1 — Git, GitHub & Exploratory Data Analysis

### ✔️ Deliverables
- GitHub repo initialized  
- `task-1` branch created and committed with EDA work  
- Data understanding: structure, types, ranges  
- Missing values, duplicates, and outlier checks  
- Univariate & bivariate analysis  
- **3+ meaningful plots**, including:  
  - Loss Ratio trends  
  - Claims per Province  
  - Premium vs Claims scatter  

### 🧠 Key Insights (Examples)
- Provinces with higher claim severity  
- Gender or vehicle-type risk differences  
- Trends and anomalies in premium distribution  

---

# ✅ Task 2 — Data Version Control (DVC)

### ✔️ What Was Implemented
- `dvc init`  
- Local DVC remote configured  
- Raw dataset added and tracked with DVC  
- Pushed to remote storage  
- DVC pipeline file (`dvc.yaml`) created  

