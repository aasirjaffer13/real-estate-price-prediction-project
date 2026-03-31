# Melbourne Real Estate Price Prediction

**Author:** AASIR JAFFER LONE  
**College:** VIT BHOPAL CSE AI/ML  
**Faculty:** BANDLA PAWAN BABU  
**Subject:** FUNDAMENTALS IN AI/ML  

---

## 📌 Project Overview
This project implements a comprehensive machine learning workflow to predict housing prices in Melbourne using the Kaggle Melbourne Housing dataset. The goal is to build a high-performance regression model that accurately estimates property values based on historical data.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `pandas` (Data manipulation)
    * `numpy` (Numerical computations)
    * `scikit-learn` (Machine Learning models and evaluation)
    * `matplotlib` (Data visualization)
* **Environment:** Jupyter Notebook / Python IDE

### 🔹 Models Used

#### ✅ Decision Tree Regressor (Baseline)
- Used as initial model  
- Evaluated using Mean Absolute Error (MAE)  

#### ⚙️ Hyperparameter Tuning
- Parameter tuned: `max_leaf_nodes`  
- Values tested: `[5, 25, 50, 100, 250, 500]`  
- Best value selected based on lowest MAE  

#### 🌲 Random Forest Regressor (Final Model)
- Ensemble technique  
- Combines multiple decision trees  
- Reduces overfitting and improves accuracy  

---

### 🧩 Workflow Diagram
      Dataset (melb_data.csv)
                ↓
      Data Preprocessing
                ↓
     Feature & Target Split
                ↓
       Train/Test Split
         ↓         ↓
 Decision Tree   Random Forest
     ↓                ↓
  MAE Score      MAE Score
                ↓
        Final Model Selected 

---

## 📂 Repository Structure
📦 Real-Estate-Price-Prediction
│
├── README.md
├── MODEL.py
├── Real Estate Price Prediction model.ipynb
├── melb_data.csv


## 🏗 Model Architecture
The project follows a structured ML pipeline:
1. **Data Preprocessing:** Handling missing values and feature selection.
2. **Baseline Model:** A **Decision Tree Regressor** to establish a performance benchmark.
3. **Hyperparameter Tuning:** Optimizing the Decision Tree by testing multiple `max_leaf_nodes` (5, 25, 50, 100, 250, 500) to find the ideal balance between bias and variance.
4. **Ensemble Model:** Implementing a **Random Forest Regressor** (100 trees) to reduce variance and improve prediction stability.

## 📊 Dataset Features
The model utilizes the following features for training:
* **Rooms, Distance, Postcode, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Lattitude, Longtitude, Propertycount.**

## 🚀 How to Clone and Run

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/real-estate-prediction.git](https://github.com/yourusername/real-estate-prediction.git)
cd real-estate-prediction 
``` 
## 📦 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```
▶️ 3. Run the Model
```bash
python MODEL.py
```
## 📊 Dataset Information
Dataset: Melbourne Housing Dataset
Target Variable: Price
Features: Automatically selected numerical columns
##🔍 Key Features
  - Automatic feature selection
  - Missing value handling
  - Train-validation split (80/20)
  - Model comparison
  - Hyperparameter tuning
  - Feature importance extraction
## 📈 Results
 ### Model	Performance
Decision Tree	- Higher MAE
Random Forest	 - ✅ Lower MAE (Best)
### 🏆 Final Model: Random Forest Regressor
Better accuracy
Reduced overfitting
Improved generalization
## 📌 Feature Importance
rf_model.feature_importances_

Helps identify the most influential features affecting house prices.

## 📚 Learning Outcomes
Supervised learning (regression)
Model comparison techniques
Hyperparameter tuning
Data preprocessing
Evaluation metrics (MAE, R²)
## ⚠️ Limitations
Only numerical features used
No advanced feature engineering
No cross-validation

## 📜 Conclusion

This project demonstrates a complete machine learning pipeline from data preprocessing to model evaluation. The Random Forest model outperforms the Decision Tree and provides better prediction accuracy.

## 🤝 Acknowledgment

Special thanks to Bandla Pawan Babu for guidance in the subject Fundamentals in AI/ML.
