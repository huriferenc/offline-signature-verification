#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classifier.classifier import CLF_TYPE

from config import CLASSIFICATION_VISUALIZATION_FNAME, CLASSIFICATION_VISUALIZATION_FOLDER, CLASSIFIER_RESULT_FNAME, CLASSIFIER_RESULTS_FOLDER, MAX_L, MIN_L

CLASSIFIER = 'linear'

USE_COLS = ['classifier', 'n', 'p', 'tn', 'fp', 'fn', 'tp']

TRAINING_SIZE = 48

TIME_INDEX = 1702853712954

def get_error_rate_dataframes(time_index: int, training_size: int, classifier: CLF_TYPE):
    FRR = np.array([])
    FAR = np.array([])
    AER = np.array([])
    
    for threshold in range(MIN_L, MAX_L + 1):
        df = pd.read_csv(CLASSIFIER_RESULTS_FOLDER / CLASSIFIER_RESULT_FNAME.format(training_size=training_size, l=threshold, time_index=time_index), usecols=USE_COLS)
        
        error_values = df[df['classifier'] == classifier][['n', 'p', 'tn', 'fp', 'fn', 'tp']]
        
        n = error_values['n']
        p = error_values['p']
        tn = error_values['tn']
        fp = error_values['fp']
        fn = error_values['fn']
        tp = error_values['tp']
        
        frr = fn / n
        far = fp / p
        aer = (frr + far) / 2
        
        FRR = np.append(FRR, frr)
        FAR = np.append(FAR, far)
        AER = np.append(AER, aer)
        
    
    return [FRR, FAR, AER]

if __name__ == "__main__":
    FRR, FAR, AER = get_error_rate_dataframes(TIME_INDEX, TRAINING_SIZE, CLASSIFIER)
    
    CLASSIFICATION_VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)
    
    df_plot = pd.DataFrame({
        'FRR': FRR,
        'FAR': FAR,
        'AER': AER
    }, index=range(MIN_L, MAX_L + 1))
    df_plot.plot.line(xlabel='Kuszobertek', ylabel='Hiba')
    
    plt.savefig(CLASSIFICATION_VISUALIZATION_FOLDER / CLASSIFICATION_VISUALIZATION_FNAME.format(classifier=CLASSIFIER, training_size=TRAINING_SIZE, time_index=TIME_INDEX))
    
    plt.show()
