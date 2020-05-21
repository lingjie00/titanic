import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#load train and test dataset
train = pd.read_csv("data/train.csv")
submission = pd.read_csv("data/test.csv")

#modify df
def clean_df(df):
    #transform names
    df["Title"] = df["Name"].str.extract('([A-Z][a-z]+\.)')

    #drop column
    df.drop(["Name","Cabin","Ticket"],axis=1,inplace=True)
    return df

train_cleaned = clean_df(train)
submission_cleaned = clean_df(submission)

#column transformer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False)
enc = OneHotEncoder(handle_unknown='ignore')
cat_pipe = Pipeline([("Impute", imp),("Encode", enc)])

scaler = StandardScaler()
imp_num = SimpleImputer(missing_values=np.nan, strategy='median', copy=False)
num_pipe = Pipeline( [("Impute",imp_num), ("Scale", scaler)] )

ct = ColumnTransformer([("Categorical", cat_pipe, ["Sex","Embarked","Title"]),
                       ("Numeric", num_pipe, ["PassengerId", "Pclass","Age","SibSp","Parch","Fare"])],
                       remainder='drop')

#create training and validation set
train_x, test_x, train_y, test_y = train_test_split(train_cleaned.drop("Survived",axis=1),
                                                    train_cleaned["Survived"],
                                                 test_size=0.2, random_state=1)

#import models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

lr = LogisticRegression(random_state=0, solver='lbfgs')
rf = RandomForestClassifier(random_state=0, n_estimators=100)
mlp = MLPClassifier(random_state=0)
nb = GaussianNB()
knn = KNeighborsClassifier()
sgd = SGDClassifier(random_state=0)
svc = SVC(gamma="auto", random_state=0)
ridge = RidgeClassifier(random_state=0)
percep = Perceptron(random_state=0)
passive = PassiveAggressiveClassifier(random_state=0)
ada = AdaBoostClassifier(random_state=0)
process = GaussianProcessClassifier(random_state=0)
qda = QuadraticDiscriminantAnalysis()
tree = DecisionTreeClassifier(random_state=0)

models = [lr,rf, mlp, nb, knn, sgd, svc, ridge, percep ,passive, ada, process, qda]

#model fitting using cross validation scoring
#using accuracy as the scoring metric
def get_score(model):
    pipe = Pipeline([("Pre-processing", ct),("Fit",model)])
    score = np.mean(cross_val_score(pipe, train_x, train_y ,cv=5))
    return score

scores = pd.DataFrame()

for model in models:
    score = get_score(model)
    scores = scores.append([[type(model).__name__, score]])
scores.rename(columns={0:"Model", 1:"Training_Score"}, inplace=True)
scores.sort_values(by="Training_Score", ascending=False, inplace=True)
scores.reset_index(inplace=True, drop=True)
scores

#selecting only top 5 models
scores.head(5)

#fine tuning parameters
def tune(model):
    pipe = Pipeline([("Pre-processing", ct),("Fit",model)])
    pipe.fit(train_x, train_y)
    return model.best_estimator_

#fine tuning parameters: random forest
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf_params = [ {"n_estimators":np.arange(100,110,5).astype(int)} ]
rf_search = GridSearchCV(rf, rf_params, cv=5)
best_rf = tune(rf_search)
rf_search.best_score_

#fine tuning parameters: SVC
svc = SVC(gamma="auto", random_state=0)
svc_params = [ {"kernel":["rbf","linear","poly","sigmoid"]} ]
svc_search = GridSearchCV(svc,svc_params,cv=5)
best_svc = tune(svc_search)
svc_search.best_score_

#fine tuning parameters: Ridge
ridge = RidgeClassifier(random_state=0)
ridge_params = [ {"alpha":np.arange(1,1000,5).astype(float)}, {} ]
ridge_search = GridSearchCV(ridge, ridge_params, cv=5)
best_ridge = tune(ridge_search)
ridge_search.best_score_

#fine tuning parameters: log regression
lr = LogisticRegression(random_state=0)
lr_params = [ {"penalty":["l2","none"], "solver":["lbfgs"]},
            {"penalty":["l1","l2"], "solver":["liblinear"]},]
lr_search = GridSearchCV(lr, lr_params, cv=5)
best_lr = tune(lr_search)
lr_search.best_score_

#fine tuning parameters: MLP
mlp = MLPClassifier(random_state=0)
mlp_params = [ {"activation":['identity', 'logistic', 'tanh', 'relu']} ]
mlp_search = GridSearchCV(mlp,mlp_params,cv=5)
best_mlp = tune(mlp_search)
mlp_search.best_score_

#get prediction score and predictions
def get_predict(model):
    pipe = Pipeline([("Pre-processing", ct),("Fit",model)])
    pipe.fit(train_x, train_y)
    score = pipe.score(test_x, test_y)
    predictions = pipe.predict(test_x)
    return (score, predictions)

#the best models
best_models = [best_rf, best_ridge, best_lr, best_svc, best_mlp]

best_scores = pd.DataFrame()
best_predictions = pd.DataFrame()

for model in best_models:
    score, predictions = get_predict(model)
    best_scores = best_scores.append([[type(model).__name__, score]])
    best_predictions = best_predictions.append( [[predictions]] )

best_predictions = best_predictions.T

best_scores.rename(columns={0:"Model", 1:"Test_Score"}, inplace=True)
best_scores.sort_values(by="Test_Score", ascending=False, inplace=True)
best_scores.reset_index(inplace=True, drop=True)
best_scores

#votting ensemble to decide best outcome
vote = pd.array(best_predictions.apply(lambda x: np.where(np.mean(x) > 0.5,1,0),axis=1)[0]).astype(int)

#score of votting ensemble
np.mean(vote==test_y)

## for submission

#Prediction using votting ensemble
def get_submit(model):
    pipe = Pipeline([("Pre-processing", ct),("Fit",model)])
    pipe.fit(train_cleaned.drop("Survived",axis=1), train_cleaned["Survived"])
    predictions = pipe.predict(submission_cleaned)
    return predictions

submit_vote = pd.DataFrame()

for model in best_models:
    prediction = get_submit(model)
    submit_vote = submit_vote.append( [[prediction]] )

submit_vote = submit_vote.T
vote_results = pd.array(submit_vote.apply(lambda x: np.where(np.mean(x) > 0.5,1,0),axis=1)[0]).astype(int)

## for submission

#save predictions
submit = pd.DataFrame([submission_cleaned["PassengerId"],vote_results], index=['PassengerId','Survived']).T
submit.to_csv('submission.csv', index=False)

submit.head()
