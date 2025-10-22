# 🧠 Employment Suitability Prediction using Decision Tree

## 📋 Project Overview
This project applies **Machine Learning (Decision Tree Classification)** to predict whether a job candidate is **suitable for employment** based on various personal and professional attributes.  
It uses the **IBM HR Analytics Attrition Dataset** from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attritiondataset).

---

## 🎯 Objective
To build and evaluate a Decision Tree Classifier that predicts employment suitability using attributes such as:
- Age  
- Education Level  
- Years of Experience  
- Technical Test Score  
- Interview Score  
- Previous Employment  

The target variable is **`suitable_for_employment`** (`Yes` / `No`).

---

## 🧩 Tasks Performed
### 1. Data Loading & Exploration
- Loaded dataset using `pandas`
- Performed Exploratory Data Analysis (EDA)
- Checked for missing values, data types, and feature distributions

### 2. Data Preprocessing
- Converted categorical features to numeric form using **Label Encoding**
- Split dataset into **Training (80%)** and **Testing (20%)** sets

### 3. Model Building
- Trained a **DecisionTreeClassifier** from `sklearn.tree`
- Used `criterion='entropy'` for better interpretability

### 4. Model Visualization
- Visualized the trained Decision Tree using `plot_tree()`  
- Saved as: **`employment_decision_tree.png`**

### 5. Model Testing & Prediction
- Predicted labels for test data  
- Tested model using **3 hypothetical candidate profiles**

### 6. Model Evaluation
Evaluated using:
- ✅ Accuracy Score  
- ✅ Confusion Matrix  
- ✅ Classification Report (Precision, Recall, F1-score)

---

## 📊 Results Summary

| Metric | Result |
|--------|---------|
| **Accuracy** | 84.35% |
| **Precision (Yes)** | 0.86 |
| **Recall (Yes)** | 0.98 |
| **F1-score (Yes)** | 0.91 |

### 🔍 Feature Importance
| Feature | Importance (%) |
|----------|----------------|
| Years of Experience | 56% |
| Age | 27% |
| Interview Score | 12% |
| Technical Test Score | 5% |

### 🧾 Predictions for Hypothetical Candidates
| Candidate | Predicted | Confidence |
|------------|------------|-------------|
| #1 | Yes | 89.15% |
| #2 | Yes | 79.89% |
| #3 | Yes | 58.97% |

---

## 💡 Insights
- Candidates with **more years of experience** and **higher interview scores** are most likely to be employed.
- Decision Trees offer clear interpretability for HR screening decisions.
- Model performance can be improved by tuning hyperparameters or using ensemble methods (e.g., Random Forest).

---

## 📦 Required Libraries
Make sure to install these dependencies before running the project:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run
1. Clone or download this repository  
2. Open it in **VS Code** or Jupyter Notebook  
3. Run the Python script:
   ```bash
   python employment_decision_tree.py
   ```
4. The tree visualization and feature importance chart will appear automatically.

---

## 📁 Output Files
- `employment_decision_tree.py` — main script  
- `employment_decision_tree.png` — visualized tree  
- `feature_importance_chart.png` *(optional)* — generated from matplotlib  
- Console output with performance metrics and predictions

---

## 🧩 Bonus Task
Performed **Feature Importance Analysis** to determine which variables most influence employment decisions.

---

## 🧠 Conclusion
The Decision Tree Classifier achieved an accuracy of **84.35%**, indicating strong predictive ability in determining employment suitability.  
This model provides clear, interpretable rules that can assist HR professionals in identifying top candidates effectively and transparently.

---

## 📚 References
- IBM HR Analytics Dataset — Kaggle  
- Scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)
- Matplotlib Documentation — [https://matplotlib.org](https://matplotlib.org)

---
