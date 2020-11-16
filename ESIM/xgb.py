from xgboost import XGBClassifier
import xgboost as xgb
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 11,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.8,
    'solsample_bytree': 0.8,
    'silent': 1,
    'eta': 0.3,
    'seed': 710,
    'nthread':4
}

plst = list(params.items())
dataset = loadtxt("soft.txt", delimiter=",")
X = dataset[:, :-1]
Y = dataset[:, -1]

seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

xgtrain = xgb.DMatrix(X_train, y_train)
xgval = xgb.DMatrix(X_test, y_test)

watchlist = [(xgtrain, 'train', (xgval, 'val'))]
model = xgb.train(plst, xgtrain, num_boost_round=100, evals=watchlist, early_stopping_rounds=50)

dtest = xgb.DMatrix(X_test)
preds = model.predict(dtest)
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy*100))

