# **House Prices - Advanced Regression Techniques**

## **Overview**
This project showcases advanced data analytics techniques combined with machine learning to predict house prices based on a variety of features. It highlights a systematic approach to handling real-world datasets, including cleaning, transforming, and engineering data for optimal performance with machine learning models.

---

## **Key Highlights**
- **Data Analysis and Preparation**: Demonstrated expertise in exploring, cleaning, and preprocessing complex datasets.
- **Advanced Techniques**: Applied feature engineering, correlation analysis, and missing value imputation using logical and statistical methods.
- **Machine Learning**: Trained and evaluated multiple regression models to identify the best-performing algorithm.
- **Languages and Tools**: 
  - Programming Language: Python  
  - Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  

---

## **Workflow**

### **1. Data Preprocessing**
- **Data Exploration**: Analyzed the dataset for key trends and patterns.
- **Handling Missing Values**: 
  - Replaced null values in categorical data with the most frequent value (mode).
  - Imputed numerical data using techniques such as mean substitution and inference based on related features.
- **Lot Frontage**: Estimated missing values using a linear regression model between `LotFrontage` and the square root of `LotArea`.

### **2. Feature Engineering**
- Created new composite features, such as total living space, combining internal and external square footage.
- Transformed categorical variables into numerical representations using one-hot encoding for better model compatibility.

### **3. Correlation Analysis**
- Conducted correlation analysis to identify features strongly associated with the target variable (`SalePrice`).
- Visualized relationships using heatmaps to guide feature selection and model improvement.

### **4. Data Splitting**
- Split the dataset into training and testing subsets:
  - **Training Dataset**: Used for model development and validation.
  - **Testing Dataset**: Evaluated model performance on unseen data.

---

## **Machine Learning Models**
Explored a variety of machine learning algorithms:
- **Primary Model**: Ridge Regression, selected for its high performance and ability to handle multicollinearity.
- **Other Models Tested**:
  - Lasso Regression
  - Random Forest Regressor
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machines

### **Model Evaluation**
- **Metrics**: Evaluated models based on accuracy and R² score on both training and validation datasets.
- **Best Model Performance**:
  - Training Accuracy: **90.1%**
  - Validation Accuracy: **87.9%**

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries and Frameworks**:
  - **Pandas**: Data manipulation and preprocessing.
  - **NumPy**: Numerical computations.
  - **Matplotlib & Seaborn**: Data visualization.
  - **Scikit-learn**: Machine learning and model evaluation.

---

## **Sample Output**
Predicted house prices for sample properties:

| **Id**  | **SalePrice**     |
|---------|-------------------|
| 1461    | 107,530.36        |
| 1462    | 162,431.23        |
| 1463    | 163,538.91        |
| 1464    | 188,432.57        |

---

## **View the Notebook**
You can view the complete notebook [here](https://colab.research.google.com/github/utkarsh201314/House-Prices-Advanced-Regression-Techniques/blob/main/House_Prices_Advanced_Regression_Techniques.ipynb).

---

## **Conclusion**
This project demonstrates:
- **Mastery of Data Analytics**: Proficient handling of messy datasets with missing values and categorical complexities.
- **Feature Engineering Expertise**: Developed impactful features to boost model performance.
- **Machine Learning Skills**: Rigorous model evaluation and selection based on performance metrics.

---

## **Future Work**
- Experiment with ensemble models like Gradient Boosting and XGBoost for improved predictions.
- Incorporate log transformations and feature scaling for better feature representation.
- Optimize hyperparameters to further enhance model performance.
