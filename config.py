#!/usr/bin/env python
from pathlib import Path

MIN_L = 2
MAX_L = 10

PERSON_ID = 1
IMAGE_NUMBER = 24

SIGN_FOLDER_ORIG = Path.cwd() / 'CEDAR' / 'full_org'
SIGN_FOLDER_FORG = Path.cwd() / 'CEDAR' / 'full_forg'

SAVING_FOLDER_ORIG = Path.cwd() / 'quasi_straight_line_segments' / f'person-{PERSON_ID}' / 'org'
SAVING_FOLDER_FORG = Path.cwd() / 'quasi_straight_line_segments' / f'person-{PERSON_ID}' / 'forg'

EXTRACTED_FEATURE_SET_SAVING_FOLDER = Path.cwd() / 'extracted_feature_set' / f'person-{PERSON_ID}'

VISUALIZATION_FOLDER = Path.cwd() / 'visualizations' / f'person-{PERSON_ID}'

CLASSIFIER_RESULTS_FOLDER = Path.cwd() / 'classifier_results' / f'person-{PERSON_ID}'

CLASSIFICATION_VISUALIZATION_FOLDER = Path.cwd() / 'classifier_visualizations' / f'person-{PERSON_ID}'

FNAME_ORIG = f'original_{PERSON_ID}' + '_{sign_index}.png'
FNAME_FORG = f'forgeries_{PERSON_ID}' + '_{sign_index}.png'

DATAFRAME_FNAME_ORIG = f'original_{PERSON_ID}' + '_{sign_index}.png.L_{l}.csv'
DATAFRAME_FNAME_FORG = f'forgeries_{PERSON_ID}' + '_{sign_index}.png.L_{l}.csv'

VISUALIZATION_FNAME_ORIG_ORIG = f'person_{PERSON_ID}' + '_orig_{sign_1}_orig_{sign_2}_{column}_L_{l}.png'
VISUALIZATION_FNAME__ORIG_FORG = f'person_{PERSON_ID}' + '_orig_{sign_1}_forg_{sign_2}_{column}_L_{l}.png'

CLASSIFIER_RESULT_FNAME = f'result_person_{PERSON_ID}' + '_training_size_{training_size}x{training_size}_L_{l}_{time_index}.csv'
CLASSIFIER_RESULT_FNAME_TEXT = f'result_person_{PERSON_ID}' + '_training_size_{training_size}x{training_size}_L_{l}_{time_index}.txt'

CLASSIFICATION_VISUALIZATION_FNAME = f'classification_person_{PERSON_ID}' + '_training_size_{training_size}x{training_size}_{classifier}_{time_index}.png'