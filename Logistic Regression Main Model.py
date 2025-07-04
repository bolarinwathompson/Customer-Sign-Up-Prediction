# Logistic Regression - ABC Grocery Task

# Import Required Packages
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# -------------------------------------------------------------------
# Import Sample Data
data_for_model = pickle.load(open("abc_classification_modelling.p", "rb"))

# -------------------------------------------------------------------
# Drop Unnecessary Columns
data_for_model.drop("customer_id", axis=1, inplace=True)

# -------------------------------------------------------------------
# Shuffle the Data
data_for_model = shuffle(data_for_model, random_state=42)

# Class Balance
data_for_model["signup_flag"].value_counts(normalize=True)

# -------------------------------------------------------------------
# Deal with Missing Data
print("Missing values per column:")
print(data_for_model.isna().sum())
data_for_model.dropna(how="any", inplace=True)

# -------------------------------------------------------------------
# Deal with Outliers using a Boxplot Approach
outlier_columns = ["distance_from_store", "total_sales", "total_items"]
for column in outlier_columns:
    lower_quantile = data_for_model[column].quantile(0.25)
    upper_quantile = data_for_model[column].quantile(0.75)
    iqr = upper_quantile - lower_quantile
    iqr_extended = iqr * 2
    min_border = lower_quantile - iqr_extended 
    max_border = upper_quantile + iqr_extended 
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    data_for_model.drop(outliers, inplace=True)
    print(f"After dropping outliers for {column}, data_for_model shape: {data_for_model.shape}")

# -------------------------------------------------------------------
# Split Input Variables and Output Variable
X = data_for_model.drop(["signup_flag"], axis=1)
y = data_for_model["signup_flag"]

# -------------------------------------------------------------------
# Split out Training and Test Sets (this step ensures X_test is defined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)

# -------------------------------------------------------------------
# Feature Selection: One-Hot Encoding for Categorical Variable(s)
categorical_vars = ["gender"]

# Use sparse_output=False for scikit-learn 1.2+; change to sparse=False if using an earlier version.
one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

# Fit the encoder on the training set's categorical column and transform both training and test sets.
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get the encoded feature names using the new method.
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert the encoded arrays into DataFrames (keeping the original indices for proper alignment).
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder_feature_names, index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder_feature_names, index=X_test.index)

# Remove the original categorical column(s) from X_train and X_test.
X_train_clean = X_train.drop(columns=categorical_vars)
X_test_clean = X_test.drop(columns=categorical_vars)

# Concatenate the cleaned DataFrames with their corresponding encoded DataFrames.
X_train = pd.concat([X_train_clean, X_train_encoded_df], axis=1)
X_test = pd.concat([X_test_clean, X_test_encoded_df], axis=1)

# Feature Selection


clf = LogisticRegression(random_state = 42, max_iter = 1000)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train, y_train)

optimal_feature_count = feature_selector.n_features_
print(f"optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

fit_grid_scores_ = feature_selector.cv_results_['mean_test_score']
plt.plot(range(1, len(fit_grid_scores_) + 1), fit_grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection Using RFE\nOptimal number of Features is {optimal_feature_count}")
plt.tight_layout()
plt.show()


clf = LogisticRegression(random_state = 42, max_iter = 1000)
clf.fit(X_train, y_train)



# Predict The Test Score
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

import numpy as np 
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix): 
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()


# Accuracy (the number of correct clasification out of all attempted classifications)

accuracy_score(y_test, y_pred_class)

# Precision of all(of all observations that were predicted as positive, how many were actually posiitve)

precision_score(y_test, y_pred_class)


# Recall (Of all Positive observations, how many did we predict as positive)

recall_score(y_test, y_pred_class)

# Fi-Score(the harmonic mean of precision and recall)

f1_score(y_test, y_pred_class)


# --------------------------------------------
# Finding the Optimal Classification Threshold
# --------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

# Define thresholds from 0 to 1 in 0.01 steps
thresholds = np.arange(0, 1.01, 0.01)

# Initialize lists to store scores
precision_scores = []
recall_scores = []
f1_scores = []

# Loop through thresholds and calculate scores
for threshold in thresholds:
    pred_class = (y_pred_prob >= threshold).astype(int)
    
    precision = precision_score(y_test, pred_class, zero_division=0)
    recall = recall_score(y_test, pred_class, zero_division=0)
    f1 = f1_score(y_test, pred_class, zero_division=0)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Identify the optimal threshold based on maximum F1 score
max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)
optimal_threshold = thresholds[max_f1_idx]

# --------------------------------------------
# Plot Precision, Recall, and F1 vs Threshold
# --------------------------------------------

plt.style.use("seaborn")
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, label="Precision", linestyle="--")
plt.plot(thresholds, recall_scores, label="Recall", linestyle="--")
plt.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)

# Highlight optimal threshold
plt.axvline(optimal_threshold, color='red', linestyle=':', label=f"Optimal Threshold = {optimal_threshold:.2f}")

plt.title(f"Optimal Threshold for Logistic Regression\nMax F1: {max_f1:.2f} at Threshold = {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

optimal_threshold = 0.44
y_pred_class_opt_thresh = (y_pred_prob >= optimal_threshold) * 1


































