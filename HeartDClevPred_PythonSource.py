# %%
# Tyler Boudreau

# Import Required packages as needed throughout
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# Location of Dataset must be set
df = pd.read_csv('C:\\Users\\Tyler\\Downloads\\Heart_disease_cleveland_new.csv')
print(df)
df.head(10)


# %%
# Create Correlation Matrix to check for Collinearity
CorrMatrix1 = df.corr()

sb.heatmap(CorrMatrix1, cmap="coolwarm", annot=True, fmt=".1g")
plt.show()

# %%
X=df.iloc[:,0:13]
X
y=df['target']

# Create Supervised Train and Unsupervised Test Partitions
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33, random_state=3573)

# Count number 0 and 1 prediction values for Heart Disease, 0 being absent, 1 being present
y.value_counts()

def plot_roc_curve(true_y,y_predt):
    fpr, tpr, thresholds = roc_curve(true_y,y_predt)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")


# %%
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logModel=LogisticRegression(max_iter=10000)
logModel.fit(X_train, y_train)
pred_y = logModel.predict(X_test)
from sklearn.metrics import accuracy_score
print('Logistic Regression Model Accuracy: {0:0.4f}'.format(accuracy_score(y_test,pred_y)*100),"%")
predresult1 = pd.DataFrame({"Actual" : y_test, "Predicted" : pred_y})
print(predresult1)

plot_roc_curve(y_test,pred_y)
print(f"Logistic Regression AUC Score: {roc_auc_score(y_test,pred_y)}")

# %%

# Testing Logistic Regression model on example data
XTestValues1 = pd.DataFrame(np.array([[63,1,0,145,233,1,2,150,0,2.3,2,0,2]]), columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
XTestValues2 = pd.DataFrame(np.array([[67,1,3,160,286,0,2,108,1,1.5,1,3,1]]), columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
Logmodelprediction1 = logModel.predict(XTestValues1)
Logmodelprediction2 = logModel.predict(XTestValues2)
print(Logmodelprediction1)
print(Logmodelprediction2)


# %%
# LightGBM Model
import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_pred, y_test)
print('LightGBM Model Accuracy: {0:0.4f}'.format(accuracy_score(y_test, y_pred)*100),"%")
predresult2 = pd.DataFrame({"Actual" : y_test, "Predicted" : y_pred})
print(predresult2)

LightGBMPred1 = clf.predict(XTestValues1)
LightGBMPred2 = clf.predict(XTestValues2)

print(LightGBMPred1)
print(LightGBMPred2)

plot_roc_curve(y_test,y_pred)
print(f"LightGBM Model AUC Score: {roc_auc_score(y_test,y_pred)}")

# %%
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred8 = clf.predict(X_test)
 # metrics are used to find accuracy or error
from sklearn import metrics 
print()

# using metrics module for accuracy calculation
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred8)*100,"%")

predresult3 = pd.DataFrame({"Actual" : y_test, "Predicted" : y_pred8})
print(predresult3)
RandomForestPred1 = clf.predict(XTestValues1)
RandomForestPred2 = clf.predict(XTestValues2)

print(RandomForestPred1)
print(RandomForestPred2)


plot_roc_curve(y_test,y_pred8)
print(f"Random Forest AUC Score: {roc_auc_score(y_test,y_pred8)}")


# %%
# ExtraTree Model
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_split=2,min_weight_fraction_leaf =0.0,n_jobs=-1)
clf.fit(X_train, y_train)
print("ExtraTree Classifier Accuracy:",clf.score(X_test, y_test)*100,"%")
y_pred9 = clf.predict(X_test)
predresult4 = pd.DataFrame({"Actual" : y_test, "Predicted" : y_pred9})
print(predresult4)
ExtraTreePred1 = clf.predict(XTestValues1)
ExtraTreePred2 = clf.predict(XTestValues2)

print(ExtraTreePred1)
print(ExtraTreePred2)

plot_roc_curve(y_test,y_pred9)
print(f"ExtraTree AUC Score: {roc_auc_score(y_test,y_pred9)}")

# %%
# XGBoost Model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred1)
print("XGBoost Accuracy:",accuracy*100,"%")
predresult5 = pd.DataFrame({"Actual" : y_test, "Predicted" : y_pred1})
print(predresult5)
XGBoostPred1 = model.predict(XTestValues1)
XGBoostPred2 = model.predict(XTestValues2)

print(XGBoostPred1)
print(XGBoostPred2)

plot_roc_curve(y_test,y_pred1)
print(f"XGBoost Model AUC Score: {roc_auc_score(y_test,y_pred1)}")

# %%
# Setup TensorFLow Model
from tensorflow.keras.models import Sequential #Helps to create Forward and backward propogation
from tensorflow.keras.layers import Dense #Helps to create neurons in ANN

# %%
# Continue TensorFlow Setup
classifier=Sequential()
classifier.add(Dense(units=11,activation='relu'))
classifier.add(Dense(units=7,activation='relu'))
classifier.add(Dense(units=6,activation='relu'))
## Adding the output layer
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])
#classifier.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])

# %%
# TensorFlow continued setup
import tensorflow as tf
early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)


# %%
# Runs TensorFlow model up to 1000 iterations or until optimal value is found
model_history=classifier.fit(X_train,y_train,validation_split=0.30,batch_size=10,epochs=1000,callbacks=early_stopping)



# %%
# Make predictions with model on test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # If greater than .5 then model returns True or present for HeartDisease


# %%
# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
print("TensorFLow Accuracy:",score*100,"%")

TensorFlowPred1 = classifier.predict(XTestValues1)
TensorFlowPred2 = classifier.predict(XTestValues2)

plot_roc_curve(y_test,y_pred)
print(f"TensorFlow AUC Score: {roc_auc_score(y_test,y_pred)}")

# %%
# Make predictions with example test values
print(TensorFlowPred1) 
print(TensorFlowPred2)

# %%
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Define K-Fold Cross Validation
#cv = RepeatedKFold(n_splits=203,n_repeats=3,random_state=1)

# Define predictor and target variables
X = df[["age",'sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = df["target"]

# Linear Regression
LinearModel1 = LinearRegression().fit(X_train, y_train)

LinearModel1.predict(X_test)
accuracy = LinearModel1.score(X_test,y_test)

print('The predicted accuracy for Linear Regression is: {0:0.4f}'.format((accuracy*100)),"%\n")

# Ridge Model
RidgeModel1 = Ridge(alpha=10)

RidgeModel1.fit(X_train,y_train)

accuracy = RidgeModel1.score(X_test,y_test)

print('The Predicted accuracy for the Ridge Model is: {0:0.4f}'.format((accuracy*100)),"%\n")


