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
