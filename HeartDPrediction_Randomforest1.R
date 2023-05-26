#Required Packages to run R code
library(rpart)
library(randomForest)
library(caret)
library(xgboost)

# File Location of Example Dataset Required to be set
Clev_HeartD <- read.csv('C:\\Users\\Tyler\\Downloads\\Heart_disease_cleveland_new.csv') # File locatiom for Data
suppressWarnings({ #supresses warnings
set.seed(3573) # Set Seed if you wish to get same results everytime
sample1 <- sample(c(TRUE,FALSE),nrow(Clev_HeartD),replace=TRUE, prob = c(0.7,0.3))
train <- Clev_HeartD[sample1, ] #Train supervised learning Partition
test <- Clev_HeartD[!sample1, ] #Test Unsupervised Learning Partition

#Function that takes all info collected from user to make prediction with model
HeartDProbabilityCalc <- function(age,sex,cp,trestbps,chol,fbs ,restecg,thalach,exang,oldpeak,slope,ca,thal) 
  {
  # Creates Data frame to store data
  HeartDTest1 <- data.frame(age = c(age), sex = c(sex), cp = c(cp),trestbps = c(trestbps),chol = c(chol),fbs = c(fbs),restecg = c(restecg),thalach = c(thalach),exang = c(exang),oldpeak = c(oldpeak),slope = c(slope),ca = c(ca),thal = c(thal))
  #Makes Prediction using values collected and RandForest model
  HeartDprediction <- predict(HeartDForest, HeartDTest1)
  #Converts predicted value to percentage to understand models predicted value
  # Values closer to 0(%) or 1 (100%) represent more model confidence in prediction
  HeartDprediction <- (as.numeric(HeartDprediction)*100)
  
  cat("Patients Predicted Probability level for Heart Disease is", HeartDprediction,"% Using Random Forest Model\n")
  
  }

cat("Welcome to the Probability of Heart Disease Program", sep = "\n")
cat("Please enter the Patients age in years", sep = "\n")
age = as.integer(readline())
cat("Please enter the Patients sex; 1 for Male, 0 for Female\n")
sex = as.integer(readline())
cat("Please enter the Patients chest pain 0 typical angina, 1 atypical angina, 2 non- anginal pain, 3 asymptomatic (Nominal)\n")
cp = as.integer(readline())
cat("Please enter the Patients resting blood pressure in mm/HG\n")
trestbps = as.integer(readline())
cat("Please enter the Patients Serum Cholesterol in mg/dl\n")
chol = as.integer(readline())
cat("Please enter the Patients Blood sugar levels on fasting > 120 mg/dl represents as 1 in case of true and 0 as false (Nominal)\n")
fbs = as.integer(readline())
cat("Please enter the Patients Result of electrocardiogram while at rest are represented in 3 distinct values
0 : Normal 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of >
0.05 mV) 2: showing probable or definite left ventricular hypertrophyby Estes' criteria (Nominal)\n")
restecg = as.integer(readline())
cat("Please enter the Patients Maimum heart Rate achieved\n")
thalach = as.integer(readline())
cat("Please enter the Patients Angina induced by exercise; 0 depicting NO, 1 depicting YES.\n")
exang = as.integer(readline())
cat("Please enter the Patients Exercise induced ST-depression in relative with state of rest\n")
oldpeak = as.numeric(readline())
cat("Please enter the Patients ST segment measured in terms of slope during peak exercise 0: up sloping; 1: flat; 2: down sloping (Nominal)\n")
slope = as.integer(readline())
cat("Please enter the Patients number of major vessels (0-3) (Nominal)\n")
ca = as.integer(readline())
cat("Please enter the Patients thalassemia level:
0: NULL, 1: normal blood flow, 2: fixed defect (no blood flow in some part of the heart), 3: reversible defect (a blood flow is observed but it is not normal(nominal)\n")
thal = as.integer(readline())

#Calls function with parameters collected
HeartDProbabilityCalc(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
}
)
