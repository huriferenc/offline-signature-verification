#!/usr/bin/env python
from datetime import datetime
import numpy as np
import pandas as pd

from edge_processor import EdgeProcessor

from config import EXTRACTED_FEATURE_SET_SAVING_FOLDER, FNAME_FORG, FNAME_ORIG, IMAGE_NUMBER, MAX_L, MIN_L, SIGN_FOLDER_FORG, SIGN_FOLDER_ORIG, SAVING_FOLDER_FORG, SAVING_FOLDER_ORIG

from params import C_NAMES, C_NUM, SEGMENT, START_SEGMENT


class App:
    def process_edges(self, l, filename, folder, savingfolder):
        edgeProcessor = EdgeProcessor(l, filename, folder, savingfolder)

        edgeProcessor.start()

        E = edgeProcessor.E

        class_segments = edgeProcessor.class_segments
        P = len(E)

        return [class_segments, P]

    def getDataFrame(self, class_segments, P):
        data_table = {
            "Class(Ci)": C_NAMES,
            "ni": [],
            "pi/P": [],
            "pi/ni": [],
            "cp/P": [],
            "mCi": [],
            "rij/P": [],
            "lRj": [[] for _ in range(C_NUM)]
        }

        max_Ci_in_regions = [
            {"Ci": 0, "count": 0},
            {"Ci": 0, "count": 0},
            {"Ci": 0, "count": 0},
            {"Ci": 0, "count": 0},
            {"Ci": 0, "count": 0},
            {"Ci": 0, "count": 0}
        ]

        for i, item in enumerate(class_segments):
            DQSSi, ni, pi, DQSSi_img = item
            
            # Ha nincs adott osztályú szegmens, akkor folytassa a következő osztállyal
            if ni == 0:
                data_table['ni'].append(0)
                data_table['pi/P'].append(0)
                data_table['pi/ni'].append(0)
                data_table['cp/P'].append(0)
                data_table['mCi'].append(0)
                data_table['rij/P'].append(0)
                continue

            # cp - Közös pixelek száma a szomszédos osztállyal
            # Ci and Cj, j = (i+1) mod 12
            _, _, _, DQSSi_neighbour_img = class_segments[(i+1) % C_NUM]
            same_pixel_condition = np.logical_and(np.logical_or(DQSSi_img == START_SEGMENT, DQSSi_img == SEGMENT), np.logical_or(
                DQSSi_neighbour_img == START_SEGMENT, DQSSi_neighbour_img == SEGMENT))
            cp = np.sum(same_pixel_condition)

            # mCi - Annak a régiónak az azonosítója, ahol a legtöbb Ci oszályú pixel létezik
            # Regions
            height, width = DQSSi_img.shape[:2]
            part_height = height // 2
            part_width = width // 3
            regions = [
                DQSSi_img[0:part_height, 0:part_width],
                DQSSi_img[0:part_height, part_width:2*part_width],
                DQSSi_img[0:part_height, 2*part_width:3*part_width],
                DQSSi_img[part_height:2*part_height, 0:part_width],
                DQSSi_img[part_height:2 *
                          part_height, part_width:2*part_width],
                DQSSi_img[part_height:2 *
                          part_height, 2*part_width:3*part_width]
            ]

            mCi = 0
            rij = 0
            for index, region in enumerate(regions):
                pixels_count = np.sum(np.logical_or(
                    region == START_SEGMENT, region == SEGMENT))

                if pixels_count > rij:
                    rij = pixels_count
                    mCi = index

                if pixels_count > max_Ci_in_regions[index]["count"]:
                    max_Ci_in_regions[index]["count"] = pixels_count
                    max_Ci_in_regions[index]["Ci"] = i

            data_table['ni'].append(ni)
            data_table['pi/P'].append(pi/P)
            data_table['pi/ni'].append(pi/ni)
            data_table['cp/P'].append(cp/P)
            data_table['mCi'].append(mCi+1)
            data_table['rij/P'].append(rij/P)

        for region_index, max_Ci in enumerate(max_Ci_in_regions):
            data_table['lRj'][max_Ci["Ci"]].append(region_index + 1)

        return pd.DataFrame(data_table)

    def export_feature_set(self, filename, savingfolder, df):
        export_path = savingfolder / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(export_path, index=False)

    def run(self):
        start_time = datetime.now()
        print(f"Starting time: {start_time}")

        saving_folder_original = SAVING_FOLDER_ORIG
        saving_folder_forgery = SAVING_FOLDER_FORG

        efs_saving_folder = EXTRACTED_FEATURE_SET_SAVING_FOLDER

        for sign_id in range(IMAGE_NUMBER):
            print(f'Sign number: {sign_id + 1}')

            filename_original = FNAME_ORIG.format(sign_index=sign_id + 1)
            filename_forgery = FNAME_FORG.format(sign_index=sign_id + 1)

            for l in range(MIN_L, MAX_L + 1):
                print(f'Threshold: {l}')

                class_segments_original, P_original = self.process_edges(l, filename_original, SIGN_FOLDER_ORIG, saving_folder_original)
                df_original = self.getDataFrame(class_segments_original, P_original)
                self.export_feature_set(f'{filename_original}.L_{l}.csv', efs_saving_folder, df_original)

                class_segments_forgery, P_forgery = self.process_edges(l, filename_forgery, SIGN_FOLDER_FORG, saving_folder_forgery)
                df_forgery = self.getDataFrame(class_segments_forgery, P_forgery)
                self.export_feature_set(f'{filename_forgery}.L_{l}.csv', efs_saving_folder, df_forgery)

        end_time = datetime.now()
        print(f"Ending time: {end_time}")
        print(f"Time difference: {end_time - start_time}")


if __name__ == "__main__":
    App().run()
