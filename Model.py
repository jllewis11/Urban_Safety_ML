from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

class Model:
    def __init__(_self,modelName,train_X,train_Y,val_X=None,val_Y=None):
        #train
        _self.train_X=train_X
        _self.train_Y=train_Y
        #val used to tune hyperparameters
        _self.val_X=val_X
        _self.val_Y=val_Y
        _self.modelName=modelName
        if(_self.modelName=="KNeighborsClassifier"): _self.model=KNeighborsClassifier()
        elif(_self.modelName=="DecisionTreeClassifier"): _self.model=DecisionTreeClassifier()
        elif(_self.modelName=="MLPClassifier"): _self.model=MLPClassifier(max_iter=1000,early_stopping=True)
        
    def train(_self, kFold, param_grid=[]):
        #use validation set to find the best hyperparameters.
        model=GridSearchCV(estimator=_self.model,param_grid=param_grid,refit=True,scoring="f1_macro",cv=kFold)
        model.fit(_self.val_X,_self.val_Y)
        print('Best hyperparameters for {model}: {hyperparams}'.format(model=_self.modelName,hyperparams=model.best_params_))
        #apply the best paraments to model
        _self.model=_self.model.set_params(**model.best_params_)
        trainRes=_self.kFoldValidation(kFold)
        _self.model.fit(pd.concat([_self.train_X,_self.val_X]),pd.concat([_self.train_Y,_self.val_Y]))
        #return train scores - acc & F1 macro
        return trainRes
        
        
    def kFoldValidation(_self,kFold):
        res=cross_validate(estimator=_self.model,
                            X=_self.train_X,
                            y=_self.train_Y,
                            cv=kFold,
                            n_jobs=-1,
                            scoring=["accuracy","f1_macro"],
                            return_estimator=True,
                            return_train_score=True)

        avgTrainACC=np.mean(res["train_accuracy"])
        avgTrainF1Macro=np.mean(res["train_f1_macro"])
        print('{kFold} Fold Validation Averaged Accuracy {avgTrainACC} and F1 Marco {avgTrainF1Macro} for {model}'.format(kFold=kFold,model=_self.modelName,avgTrainACC=avgTrainACC,avgTrainF1Macro=avgTrainF1Macro))
        return {"trainACC":avgTrainACC,"trainF1Macro":avgTrainF1Macro}

    def test(_self,test_X, test_Y):
        Y_hat=_self.predict(test_X)
        testACC=accuracy_score(test_Y,Y_hat)
        testF1Macro=f1_score(test_Y,Y_hat,average="macro")
        print('Test Accuracy {testACC} and F1 Marco {testF1Macro} for {model}'.format(model=_self.modelName,testACC=testACC,testF1Macro=testF1Macro))
        return {"testACC":testACC, "testF1Macro":testF1Macro}
        
        
    def predict(_self, x):
        return _self.model.predict(x)