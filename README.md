# ABC Grocery Customer Sign-Up Prediction using Logistic Regression 

## Project Overview:
The **ABC Grocery Logistic Regression** project uses **Logistic Regression** to predict whether a customer will sign up for a promotional offer based on customer demographics and transaction data. The model helps ABC Grocery identify high-potential customers and optimize marketing efforts. The project involves multiple steps, including **feature selection**, **cross-validation**, and **model evaluation**.

## Objective:
The goal of this project is to build a **Logistic Regression model** that can predict the likelihood of a customer signing up for a promotional offer based on various factors, including **purchase behavior**, **customer demographics**, and **transaction history**. The model will help ABC Grocery target customers more effectively, improving the return on marketing campaigns.

## Key Features:
- **Data Preprocessing**: Raw data is cleaned and prepared for modeling, including handling missing values, encoding categorical variables, and outlier removal.
- **Feature Selection**: **Recursive Feature Elimination with Cross-Validation (RFECV)** is used to select the most relevant features that contribute to predicting customer sign-ups.
- **Model Training**: The **Logistic Regression model** is trained using the cleaned and preprocessed data.
- **Model Evaluation**: The model’s performance is evaluated using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and a **confusion matrix**.
- **Threshold Optimization**: The model's classification threshold is optimized to balance precision and recall, based on **F1 score**.

## Methods & Techniques:

### **1. Data Preprocessing**:
- The dataset is cleaned and any missing values are handled by removing or imputing them.
- **One-Hot Encoding** is applied to convert categorical variables like **gender** into numerical features that can be used for modeling.
- **Outlier Detection and Removal**: Outliers in key features, such as **distance from store** and **total sales**, are detected using the **Interquartile Range (IQR)** method and removed from the dataset to improve model performance.

### **2. Feature Selection with RFECV**:
To improve model performance and reduce overfitting, **Recursive Feature Elimination with Cross-Validation (RFECV)** is used. This technique helps in selecting the most important features by iteratively removing less relevant ones.

### **3. Logistic Regression Model**:
- A **Logistic Regression** model is trained on the preprocessed dataset. The model predicts the likelihood of customer sign-up based on features such as **total sales**, **distance from the store**, and **total items purchased**.
- **Cross-validation** is used to assess the model’s robustness and prevent overfitting.

### **4. Model Evaluation**:
- **Accuracy**: Measures the proportion of correct predictions.
- **Precision**: Indicates the percentage of positive predictions that are actually correct.
- **Recall**: Represents the percentage of actual positive cases correctly predicted.
- **F1-score**: The harmonic mean of precision and recall, providing a balance between the two metrics.
- **Confusion Matrix**: A visualization of the model’s performance, showing true positives, true negatives, false positives, and false negatives.

### **5. Threshold Optimization**:
The model’s classification threshold is fine-tuned to maximize the **F1 score**. By adjusting the threshold for classifying a customer as likely to sign up, the model’s **precision** and **recall** are balanced to ensure the best possible outcome.

### **6. Model Visualization**:
- **Confusion Matrix** is visualized to better understand model performance and identify areas where the model is making incorrect predictions.
- A plot of **precision**, **recall**, and **F1 score** against various threshold values is created to identify the optimal threshold for classification.

## Technologies Used:
- **Python**: Programming language for data manipulation, model implementation, and evaluation.
- **scikit-learn**: For implementing **Logistic Regression**, **RFECV**, **train-test split**, and evaluation metrics.
- **pandas**: For data manipulation and preprocessing.
- **matplotlib**: For visualizing model performance and metrics.
- **pickle**: For saving the trained model and allowing for easy reuse in future predictions.

## Key Results & Outcomes:
- The **Logistic Regression model** predicts customer sign-ups with high accuracy, precision, and recall, enabling ABC Grocery to effectively target potential customers.
- **Feature selection** using **RFECV** improved model interpretability and performance by focusing on the most relevant features.
- **Threshold optimization** enhanced the model’s ability to balance precision and recall, ensuring a better trade-off between false positives and false negatives.

## Lessons Learned:
- **Feature engineering** and **preprocessing** are crucial for improving the performance of Logistic Regression models.
- **Cross-validation** and **feature selection** help in building robust models that generalize well to unseen data.
- **Threshold optimization** is important for fine-tuning model predictions, especially in imbalanced classification tasks.

## Future Enhancements:
- **Model Optimization**: Further hyperparameter tuning using **GridSearchCV** or **RandomizedSearchCV** could further improve the model’s accuracy.
- **Advanced Classification Models**: Exploring other models such as **Random Forest**, **Gradient Boosting**, or **XGBoost** for potentially better results.
- **Real-Time Prediction**: Deploying the model in a real-time environment to predict customer sign-ups as new transaction data comes in.
