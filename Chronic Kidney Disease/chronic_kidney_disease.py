import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#Importing the dataset
dataset = pd.read_csv('Chronic_Kidney_Disease.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Handling of missing data
imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
imputer.fit(X)
X = imputer.transform(X)

#Handling of categorical data
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

columnTransformer = ColumnTransformer([('OneHotEncoder', OneHotEncoder(), [5, 6, 7, 8, 18, 19, 20, 21, 22, 23])], 
                                       remainder = 'passthrough')
columnTransformer.fit(X)
X = np.array(columnTransformer.transform(X), dtype = np.float)

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

#Support Vector Classifier
support_vector_classifier = SVC(kernel = 'rbf', gamma = 'auto', random_state = 0)
support_vector_classifier.fit(X_train, y_train)
y_pred_svc = support_vector_classifier.predict(X_test)

#Accuracy score using 10 fold cross validation
accuracy_svc = accuracy_score(y_test, y_pred_svc)
cross_val_svc = cross_val_score(support_vector_classifier, X, y, cv = 10)
print('Accuracy for support vector classifier = %' , accuracy_svc*100)
print(cross_val_svc)

#Confusion Matrix of SVC Model
cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.matshow(cm_svc)
plt.title('Confusion Matrix of SVC Model')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Decisioın Tree Classifier
decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy',  random_state = 0)
decision_tree_classifier.fit(X_train, y_train)
y_pred_dtc = decision_tree_classifier.predict(X_test)

#Accuracy score using 10 fold cross validation
accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
cross_val_dtc = cross_val_score(decision_tree_classifier, X, y, cv = 10)
print('Accuracy for decision tree classifier = %' , accuracy_dtc*100)
print(cross_val_dtc)

#Confusion Matrix of Decision Tree Model
cm_dtc = confusion_matrix(y_test, y_pred_dtc)
plt.matshow(cm_dtc)
plt.title('Confusion Matrix of Decision Tree Model')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print('New entry: ')
new_entry = []
new_entry.append(input('Age: '))
new_entry.append(input('Blood pressure (mm/Hg): '))
new_entry.append(input('Specific Gravity (1.005,1.010,1.015,1.020,1.025): '))
new_entry.append(input('Albumin (0,1,2,3,4,5): '))
new_entry.append(input('Sugar (0,1,2,3,4,5): '))
new_entry.append(input('Red Blood Cells (normal, abnormal): '))
new_entry.append(input('Pus Cell (normal, abnormal): '))
new_entry.append(input('Pus Cell Clumps (present, notpresent): '))
new_entry.append(input('Bacteria (present, notpresent): '))
new_entry.append(input('Blood Glucose Random (mgs/dl): '))
new_entry.append(input('Blood Urea (mgs/dl): '))
new_entry.append(input('Serum Creatinine (mgs/dl): '))
new_entry.append(input('Sodium (mEq/L): '))
new_entry.append(input('Potassium (mEq/L): '))
new_entry.append(input('Hemoglobin (gms): '))
new_entry.append(input('Packed  Cell Volume: '))
new_entry.append(input('White Blood Cell Count (cells/cumm): '))
new_entry.append(input('Red Blood Cell Count (millions/cmm): '))
new_entry.append(input('Hypertension (yes, no): '))
new_entry.append(input('Diabetes Mellitus (yes, no): '))
new_entry.append(input('Coronary Artery Disease (yes, no): '))
new_entry.append(input('Appetite (good, poor): '))
new_entry.append(input('Pedal Edema (yes, no): '))
new_entry.append(input('Anemia (yes, no): '))

new_entry = np.matrix(new_entry)

new_entry = np.array(columnTransformer.transform(new_entry), dtype = np.float)
new_entry = scale_X.transform(new_entry)
svc_pred = support_vector_classifier.predict(new_entry)
dtc_pred = decision_tree_classifier.predict(new_entry)
print("SVC Model's prediction for the new entry: " + labelEncoder.inverse_transform(svc_pred))
print("Decision Tree Model's prediction for the new entry: " + labelEncoder.inverse_transform(dtc_pred))