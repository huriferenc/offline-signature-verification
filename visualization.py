#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATAFRAME_FNAME_FORG, DATAFRAME_FNAME_ORIG, EXTRACTED_FEATURE_SET_SAVING_FOLDER, PERSON_ID, VISUALIZATION_FNAME__ORIG_FORG, VISUALIZATION_FNAME_ORIG_ORIG, VISUALIZATION_FOLDER

THRESHOLD = 4

SIGN_NUMBER_BASE = 1
SIGN_NUMBER_SECOND = 9

def visualize_data_frame_sections(df_1, df_2, df_2_forged):
    indexes = np.arange(12)
    selected_indexes = np.where((indexes == 0) | (indexes % 2 == 1))
    
    VISUALIZATION_FOLDER.mkdir(parents=True, exist_ok=True)

    for col in df_1.columns[1:7]:
        df_1_vs_df_2 = pd.DataFrame({
            f'Person-{PERSON_ID}-Ori-{SIGN_NUMBER_BASE}': np.array(df_1[col])[selected_indexes],
            f'Person-{PERSON_ID}-Ori-{SIGN_NUMBER_SECOND}': np.array(df_2[col])[selected_indexes]
        }, index=[1, 2, 4, 6, 8, 10, 12])

        df_1_vs_df_2.plot.line(xlabel='Class no. (Ci)', ylabel=col, title=f'Threshold={THRESHOLD}')
        
        plt.savefig(VISUALIZATION_FOLDER / VISUALIZATION_FNAME_ORIG_ORIG.format(column=col.replace("/", ""), sign_1=SIGN_NUMBER_BASE, sign_2=SIGN_NUMBER_SECOND, l=THRESHOLD))
        # plt.close()

        df_1_vs_df_2_forged = pd.DataFrame({
            f'Person-{PERSON_ID}-Ori-{SIGN_NUMBER_BASE}': np.array(df_1[col])[selected_indexes],
            f'Person-{PERSON_ID}-Forged-{SIGN_NUMBER_SECOND}': np.array(df_2_forged[col])[selected_indexes]
        }, index=[1, 2, 4, 6, 8, 10, 12])

        df_1_vs_df_2_forged.plot.line(xlabel='Class no. (Ci)', ylabel=col, title=f'Threshold={THRESHOLD}')
        
        plt.savefig(VISUALIZATION_FOLDER / VISUALIZATION_FNAME__ORIG_FORG.format(column=col.replace("/", ""), sign_1=SIGN_NUMBER_BASE, sign_2=SIGN_NUMBER_SECOND, l=THRESHOLD))
        # plt.close()

    plt.show()


if __name__ == "__main__":
    df_origin_1 = pd.read_csv(EXTRACTED_FEATURE_SET_SAVING_FOLDER / DATAFRAME_FNAME_ORIG.format(sign_index=SIGN_NUMBER_BASE, l=THRESHOLD))
    df_origin_9 = pd.read_csv(EXTRACTED_FEATURE_SET_SAVING_FOLDER / DATAFRAME_FNAME_ORIG.format(sign_index=SIGN_NUMBER_SECOND, l=THRESHOLD))
    df_forg_9 = pd.read_csv(EXTRACTED_FEATURE_SET_SAVING_FOLDER / DATAFRAME_FNAME_FORG.format(sign_index=SIGN_NUMBER_SECOND, l=THRESHOLD))

    visualize_data_frame_sections(df_origin_1,  df_origin_9, df_forg_9)
