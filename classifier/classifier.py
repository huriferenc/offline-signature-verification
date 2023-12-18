#!/usr/bin/env python
from datetime import datetime
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
import random
import multiprocessing

from config import DATAFRAME_FNAME_FORG, DATAFRAME_FNAME_ORIG, EXTRACTED_FEATURE_SET_SAVING_FOLDER, IMAGE_NUMBER

USE_COLS = ['ni', 'pi/P', 'pi/ni', 'cp/P', 'mCi']

K_FOLD_NUMBER = 8

KERNEL = ['linear', 'rbf', 'poly', 'sigmoid']

CLF_TYPE = Literal['linear', 'rbf', 'poly', 'sigmoid', 'random_forest']


class Classifier:
    C = np.arange(0.0001, 1.0001, 0.0001)

    gamma = np.arange(0.01, 1.01, 0.05)
    
    is_kernel = False
    
    def __init__(self, training_size: int, threshold: int, clf_type: CLF_TYPE) -> None:
      global KERNEL
      
      self.training_size = training_size
      self.threshold = threshold

      self.clf_type: CLF_TYPE = clf_type
      
      self.is_kernel = self.clf_type in KERNEL
      
      self.param_grid = None
      
      if self.is_kernel:
        self.param_grid = {
            'C': self.C,
            'kernel': [self.clf_type]
        }
        
        if self.clf_type == 'rbf':
            self.param_grid['gamma'] = self.gamma
      else:
        # RandomForest
        self.param_grid = {
            'bootstrap': [True],
            'max_depth': [3, 4, 5],
            'max_features': [2, 3, 4, 5],
            'n_estimators': [10, 25, 50]
        }
        
      self.best_params = None
      self.best_score = None
      self.best_accuracy = None
      self.best_accuracy_str: str = ""
      self.accuracy = None
      self.confusion_matrix = None
      self.total_n = 0
      self.total_p = 0
      self.tn = 0
      self.fp = 0
      self.fn = 0
      self.tp = 0
      
      self.runtime = None

    def get_result(self):
        return {
            "classifier": self.clf_type,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_accuracy": self.best_accuracy,
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "n": self.total_n,
            "p": self.total_p,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "runtime": self.runtime
        }
    
    def __str__(self) -> str:
        return f'''
##################################################################################################################
{self.clf_type.capitalize()}{' SVM' if self.is_kernel else ''} Classifier:

Legjobb parameterek: {self.best_params}
Eredmeny : {self.best_score*100}%
Legjobb hiperparameterek melletti pontossaga:
{self.best_accuracy_str}
Pontossag: {self.accuracy}
Tevesztesi matrix:
{self.confusion_matrix}
Osszes hamis alairasok szama(N): {self.total_n}
Osszes valodi alairasok szama(P): {self.total_p}
Tevesen valodinak cimkezett alairasok szama(FP): {self.fp}
Helyesen hamisnak cimkezett alairasok szama(TN): {self.tn}
Tevesen hamisnak cimkezett alairasok szama(FN): {self.fn}
Helyesen valodinak cimkezett alairasok szama(TP): {self.tp}
Futasi ido: {self.runtime}
##################################################################################################################
'''

    def get_original_signature_dfs(self, image_number, threshold, use_cols=[]):
        arr = np.array([])

        for i in range(image_number):
            df = pd.read_csv(EXTRACTED_FEATURE_SET_SAVING_FOLDER / DATAFRAME_FNAME_ORIG.format(sign_index=i+1, l=threshold), usecols=use_cols)

            arr = np.append(arr, {
                "X": df.to_numpy(),
                "y": 1
            })

        return arr


    def get_forgery_signature_dfs(self, image_number, threshold, use_cols=[]):
        arr = np.array([])

        for i in range(image_number):
            df = pd.read_csv(EXTRACTED_FEATURE_SET_SAVING_FOLDER / DATAFRAME_FNAME_FORG.format(sign_index=i+1, l=threshold), usecols=use_cols)

            arr = np.append(arr, {
                "X": df.to_numpy(),
                "y": 0
            })

        return arr

    def randomize_list(self, dfs):
        indexes = list(range(len(dfs)))

        random_indexes = np.random.choice(indexes, size=len(dfs), replace=False)

        random_list = np.array([])

        for i in random_indexes:
            random_list = np.append(random_list, dfs[i])

        return random_list

    def randomize_list_by_training_size(self, dfs, training_size: int):
        indexes = list(range(len(dfs)))

        # Véletlenszerűen választunk training_size db elemet
        random_indexes = np.random.choice(indexes, size=training_size, replace=False)

        random_list = np.array([])

        for i in random_indexes:
            random_list = np.append(random_list, dfs[i])

        return random_list
          
    def run(self):
        global USE_COLS, K_FOLD_NUMBER, IMAGE_NUMBER
        
        print('##################################################################################################################\n')
        print(f"Running {self.clf_type.capitalize()}{' SVM' if self.is_kernel else ''} Classifier...\n")
        
        start_time = datetime.now()

        ###########################################################################################
        ################################## Starting main process ##################################
        ###########################################################################################
        origin_dfs = self.get_original_signature_dfs(IMAGE_NUMBER, self.threshold, USE_COLS)
        forgery_dfs = self.get_forgery_signature_dfs(IMAGE_NUMBER, self.threshold, USE_COLS)
        dfs = np.concatenate([origin_dfs, forgery_dfs])
        # random_dfs = self.randomize_list(dfs)
        random_dfs = self.randomize_list_by_training_size(dfs, self.training_size)

        X = np.array([df['X'] for df in random_dfs])
        y = np.array([df['y'] for df in random_dfs])
        
        self.total_p = len(y[y == 1])
        self.total_n = len(y[y == 0])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1/K_FOLD_NUMBER, random_state=42, shuffle=True)
        
        # Az SVC modell alapvetően két dimenziós bemeneti adatokra van tervezve, mivel a klasszikus SVM
        # két osztály közötti határt próbál meghatározni egy kétdimenziós térben. Ha az adatkészlet három
        # vagy több dimenziót tartalmaz, akkor azt a SVC nem tudja kezelni.
        nsamples, nx, ny = X_train.shape
        X_train_reduced = X_train.reshape((nsamples,nx*ny))
        nsamples_2, nx_2, ny_2 = X_test.shape
        X_test_reduced = X_test.reshape((nsamples_2,nx_2*ny_2))

        # Elérhető CPU magok száma
        available_cores = multiprocessing.cpu_count()
        print('Available CPU cores:', available_cores)
        
        # Initialization of classifier
        clf = None
        if self.is_kernel:
            clf = svm.SVC(kernel=self.clf_type)
        else:
            clf = RandomForestClassifier(random_state=42)
        
        # Legjobb hiperparaméterek keresése a gépi tanulási modell számára
        grid_search = GridSearchCV(estimator=clf, param_grid=self.param_grid,
                            cv=K_FOLD_NUMBER, n_jobs=available_cores, verbose=3)  # n_jobs -> processzorok száma
        
        # Model tanítása és legjobb paraméterek kiválasztása
        grid_search.fit(X_train_reduced, y_train) 
        
        # A legjobb paraméterek kinyerése
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # SVM létrehozása a legjobb paraméterekkel
        best_clf = grid_search.best_estimator_
        # best_clf = None
        # if self.clf_type == 'rbf':
        #     best_clf = svm.SVC(kernel=self.clf_type, C=self.best_params['C'], gamma=self.best_params['gamma'])
        # else:
        #     best_clf = svm.SVC(kernel=self.clf_type, C=self.best_params['C'])
        
        # Model tanítása
        best_clf.fit(X_train_reduced, y_train)
            
        # Legjobb hiperparaméterek melletti pontosságának kiértékelése
        grid_predictions = grid_search.predict(X_test_reduced)
        self.best_accuracy = classification_report(y_true=y_test, y_pred=grid_predictions, output_dict=True)
        self.best_accuracy_str = classification_report(y_true=y_test, y_pred=grid_predictions, output_dict=False)
        # self.best_accuracy = classification_report(y_test, y_pred)

        # Predikciók elkészítése
        y_pred = best_clf.predict(X_test_reduced)

        # Pontosság kiértékelése
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Tévesztési mátrix generálása
        # TN  |  FP
        # FN  |  TP
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        self.tn = self.confusion_matrix[0][0]
        self.fp = self.confusion_matrix[0][1]
        self.fn = self.confusion_matrix[1][0]
        self.tp = self.confusion_matrix[1][1]
        
        ###########################################################################################
        ################################### Ending main process ###################################
        ###########################################################################################

        end_time = datetime.now()
        
        self.runtime = end_time - start_time
        
        print('\n##################################################################################################################')