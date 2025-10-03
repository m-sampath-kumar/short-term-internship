# 🚗 Engine Failure Prediction Dashboard

This repository contains a **Streamlit-based Machine Learning project** to predict engine temperature and failures using sensor data. The project demonstrates data preprocessing, model training, visualization, and interactive prediction using multiple regression models.

---

## 📌 Project Overview

The main goal of this project is to **predict engine temperature** based on sensor readings to anticipate failures and enable preventive maintenance. Users can upload their dataset, explore the data, and interact with multiple regression models to see predictions in real-time.

**Features include:**
- Upload and preview engine sensor dataset (`.csv`)  
- Explore dataset statistics (mean, min, max, etc.)  
- Train and evaluate multiple regression models:
  - Linear Regression  
  - Decision Tree Regressor  
  - Random Forest Regressor  
  - Support Vector Regressor (SVR)  
  - K-Nearest Neighbors (KNN) Regressor  
- View model performance metrics: MAE and RMSE  
- Visualize feature importance for tree-based models  
- Input custom values to get live predictions  
- Compare all regression models to identify the best-performing one  

---

## 📁 Repository Structure
engine-failure-dashboard/
│
├── app.py # Main Streamlit application

├── engine_failure_dataset.csv # Sample dataset (if available)

├── README.md # Project description

├── requirements.txt # Python dependencies

└── visualizations/ # Optional: saved plots

## 🛠️ Technologies Used
- **Python**  
- **Streamlit** for interactive web app  
- **Pandas & NumPy** for data handling  
- **Matplotlib & Seaborn** for visualization  
- **Scikit-learn** for regression models (Linear, Decision Tree, Random Forest, SVR, KNN)  


## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/engine-failure-dashboard.git
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Upload your engine_failure_dataset.csv when prompted to start predictions and visualizations.

📊 Features & Functionality
Dataset Preview: Explore uploaded CSV data and view summary statistics.

Model Training: Train selected regression models and evaluate performance.

User Input Prediction: Enter custom sensor values to predict engine temperature.

Visualizations: Feature importance plots, prediction graphs, and model comparisons.

Comparison Dashboard: Quickly see which model performs best using MAE and RMSE.

👨‍💻 Author
Mylapilli Sampath Kumar

B.Sc Computer Science (Honours), Gayatri Vidya Parishad College

GitHub: 

⚡ Goals & Learning Outcomes
Gain hands-on experience with machine learning regression models.

Learn to build interactive dashboards using Streamlit.

Understand data preprocessing, scaling, and model evaluation techniques.

Develop skills to visualize model performance and feature importance.

Feel free to explore the app, experiment with your own datasets, and provide feedback! 💻✨
