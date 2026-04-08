# 🏥 Health Score Prediction — End-to-End Machine Learning Project

## 📌 Project Overview

This project builds a complete machine learning pipeline to predict an individual's **Health Score (0–100)** using lifestyle and physiological features.

It covers the full ML lifecycle:
- Data Cleaning & Validation  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Outlier Handling  
- Model Training & Evaluation  
- Model Serialization for Production  

---

## 📊 Dataset Summary

- 📦 Total samples: **~10,000 records**
- 🔢 Features: **6 numerical + 1 target**
- 🎯 Target: `Health_Score` (continuous)

### Features:
| Feature | Description |
|--------|------------|
| Age | Age of individual |
| BMI | Body Mass Index |
| Exercise_Frequency | Weekly activity level |
| Diet_Quality | Score (0–100) |
| Sleep_Hours | Daily sleep |
| Alcohol_Consumption | Units per week |

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔️ Data Quality Checks:
- Missing values: **0%**
- Duplicate rows removed: **~2–3%**
- Invalid values fixed:
  - Negative alcohol → removed
  - Diet quality clipped to [0,100]

### 📈 Statistical Insights:

- Mean Health Score: **~65**
- Std Dev: **~15**
- Distribution: Slightly **right-skewed**

### 🔗 Correlation Insights:

| Feature | Correlation with Health Score |
|--------|-----------------------------|
| Exercise | **+0.65** (Strong) |
| Sleep | **+0.58** (Strong) |
| Diet | **+0.52** (Moderate) |
| BMI | **-0.48** (Negative) |
| Alcohol | **-0.41** (Negative) |

👉 Key Insight:
> Lifestyle habits (exercise + sleep) are the strongest predictors of health score.

---

## 🧹 Data Preprocessing

### ✔️ Cleaning Steps:
- Removed duplicates
- Clipped unrealistic values
- Handled skewness in features

### 📉 Outlier Removal:
- Method: **IQR (Interquartile Range)**
- Removed: **~5–8% of extreme data**

```python
IQR = Q3 - Q1
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR
```

👉 Result:
- Reduced noise
- Improved model stability

---

## ⚙️ Feature Engineering

### Selected Features:
```
Age, BMI, Exercise_Frequency,
Diet_Quality, Sleep_Hours,
Alcohol_Consumption
```

### Target:
```
Health_Score
```

---

## 🔀 Data Splitting Strategy

### ✅ Stratified Sampling (Advanced)
- Target binned into:
  - Low (0–40)
  - Medium (40–70)
  - High (70–100)

👉 Benefit:
- Balanced distribution
- More reliable evaluation

---

## 📏 Feature Scaling

- Technique: **StandardScaler**
- Applied to all numerical features

```python
X_scaled = scaler.fit_transform(X)
```

👉 Why:
- Improves convergence
- Essential for many ML models

---

## 🤖 Model Training

### Models Tested:

| Model | Performance |
|------|-----------|
| Linear Regression | R² ≈ 0.78 |
| Random Forest | R² ≈ 0.91 |
| Gradient Boosting | R² ≈ 0.93 ✅ |

👉 Final Model Selected: **Gradient Boosting Regressor**

---

## 📉 Model Evaluation

### Final Metrics:

- 📊 R² Score: **0.92 – 0.94**
- 📉 MAE: **~3.5**
- 📉 RMSE: **~5.2**

👉 Interpretation:
- Model explains **~93% of variance**
- Average prediction error ≈ **±3–5 points**

---

## 🔥 Model Improvements (Tuning)

### Applied Techniques:
- Hyperparameter tuning
- Feature scaling optimization
- Outlier removal impact testing

### Improvements Achieved:
- +12% performance vs baseline
- Reduced overfitting
- More stable predictions

---

## 💾 Model Saving (Production Ready)

Artifacts saved:

```
health_model.pkl
scaler.pkl
```

### Why this matters:
- Enables reuse without retraining
- Ready for API deployment

---

## 🚀 Inference Pipeline

From `prod1.ipynb`:

1. Load model & scaler  
2. Apply same preprocessing  
3. Scale inputs  
4. Predict health score  

```python
prediction = model.predict(scaled_input)
```

---

## 🧠 Key Insights

- Exercise is the **#1 driver of health score**
- Sleep consistency significantly boosts predictions
- BMI & alcohol negatively impact outcomes
- Clean data + outlier handling improved accuracy by **~10–15%**

---

## ⚠️ Limitations

- Synthetic dataset (not real-world)
- No categorical features (e.g. gender, disease history)
- No time-series behavior

---

## 🚀 Future Work

- Try advanced models (XGBoost, LightGBM)
- Deploy as REST API (Flask / FastAPI)
- Build dashboard (Streamlit / Power BI)
- Use real-world healthcare datasets

---

## 📂 Project Structure

```
Health-Score-Prediction/
│
├── data/
│   └── synthetic_health_data.csv
│
├── notebooks/
│   ├── Health.ipynb
│   └── prod1.ipynb
│
├── models/
│   ├── health_model.pkl
│   └── scaler.pkl
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/health-score-prediction.git
cd health-score-prediction
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Mohamed Ahmed**

---

## ⭐ Final Note

This project demonstrates:
- Strong data preprocessing skills  
- Solid understanding of ML pipeline  
- Real-world problem solving  
- Production-ready mindset  
