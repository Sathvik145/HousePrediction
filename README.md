
# House-Price-Prediction

## Project Overview
This project aims to predict house prices based on the California Housing dataset using various machine learning models. The best-performing model was selected after hyperparameter tuning and deployed using Flask.

---

## Dataset
The *California Housing Dataset* was used for this project. It contains information on various housing features such as median income, house age, total rooms, total bedrooms, population, households, latitude, and longitude.

---

## Data Preprocessing
- Handled missing values
- Standardized numerical features using *StandardScaler*
- Split the dataset into training (80%) and testing (20%) sets

---

## Model Training & Evaluation

### Initial Model Performance:
| Model                     | R² Score | RMSE  | MAE  |
|---------------------------|---------|--------|--------|
| Linear Regression         | 0.576   | 0.746  | 0.533  |
| Random Forest             | 0.805   | 0.505  | 0.328  |
| Decision Tree             | 0.623   | 0.703  | 0.454  |
| Gradient Boosting         | 0.776   | 0.542  | 0.372  |
| Support Vector Machine    | 0.729   | 0.596  | 0.398  |
| K-Nearest Neighbors       | 0.669   | 0.659  | 0.446  |

### Hyperparameter Tuning Results:
- *Random Forest (Tuned)*  
  - *R² Score:* 0.806  
  - *RMSE:* 0.505  
  - *MAE:* 0.327  

- *Gradient Boosting (Tuned) - Best Model*  
  - *R² Score:* 0.823  
  - *RMSE:* 0.482  
  - *MAE:* 0.323  

Based on the evaluation metrics, *Gradient Boosting Regressor* was selected as the final model.

---

## Deployment
The model was deployed using *Flask* to provide a user-friendly web interface where users can input house features and get predicted prices.

### Steps:
1. Trained the model and saved it using *joblib*
2. Built a *Flask API* to serve predictions
3. Created an interactive *HTML & CSS* interface
4. Hosted the application

---


![{491237BC-4576-457F-BFE8-EDDD2A0228B9}](https://github.com/user-attachments/assets/ef9fce98-3481-4baa-b289-e8bf3c49cf87)

![{4E0437CC-11C1-48F3-926E-8389489EC7DD}](https://github.com/user-attachments/assets/dea29b26-79c9-4979-a95c-5783d71cfffb)
