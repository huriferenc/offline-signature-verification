#!/usr/bin/env python
from datetime import datetime
import pandas as pd
from classifier.classifier import CLF_TYPE, Classifier
from config import (
    CLASSIFIER_RESULT_FNAME,
    CLASSIFIER_RESULT_FNAME_TEXT,
    CLASSIFIER_RESULTS_FOLDER,
    MAX_L,
    MIN_L,
)

TRAINING_SIZES = [48, 24]

CLASSIFIERS: list[CLF_TYPE] = ["linear", "rbf", "poly", "sigmoid", "random_forest"]

NOW_TIME_INDEX = round(datetime.now().timestamp() * 1000)


def export_results(results: list, training_size: int, threshold: int):
    data_table = {
        "classifier": [],
        "best_params": [],
        "best_score": [],
        "best_accuracy": [],
        "accuracy": [],
        "confusion_matrix": [],
        "n": [],
        "p": [],
        "tn": [],
        "fp": [],
        "fn": [],
        "tp": [],
        "runtime": [],
    }

    for result in results:
        data_table["classifier"].append(result["classifier"])
        data_table["best_params"].append(str(result["best_params"]))
        data_table["best_score"].append(result["best_score"])
        data_table["best_accuracy"].append(str(result["best_accuracy"]))
        data_table["accuracy"].append(result["accuracy"])
        data_table["confusion_matrix"].append(str(result["confusion_matrix"]))
        data_table["n"].append(result["n"])
        data_table["p"].append(result["p"])
        data_table["tn"].append(result["tn"])
        data_table["fp"].append(result["fp"])
        data_table["fn"].append(result["fn"])
        data_table["tp"].append(result["tp"])
        data_table["runtime"].append(result["runtime"])

    df = pd.DataFrame(data_table)

    # df_best_params = pd.DataFrame([self.best_params])
    # df = pd.concat([df, df_best_params], ignore_index=True)

    export_path = CLASSIFIER_RESULTS_FOLDER / CLASSIFIER_RESULT_FNAME.format(
        training_size=training_size, l=threshold, time_index=NOW_TIME_INDEX
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(export_path, index=False)


def export_results_str(results: list, training_size: int, threshold: int):
    export_path = CLASSIFIER_RESULTS_FOLDER / CLASSIFIER_RESULT_FNAME_TEXT.format(
        training_size=training_size, l=threshold, time_index=NOW_TIME_INDEX
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(export_path, "a") as file:
        for result in results:
            file.write(result + "\n")


if __name__ == "__main__":
    start_time = datetime.now()

    for threshold in range(MIN_L, MAX_L + 1):
        print(f"Threshold: {threshold}")

        for training_size in TRAINING_SIZES:
            print(f"Training size: {training_size}")

            results = []
            result_str = []

            for clf_type in CLASSIFIERS:
                print(f"Classifier: {clf_type}")

                classifier = Classifier(training_size, threshold, clf_type)

                classifier.run()

                result_str.append(str(classifier))

                results.append(classifier.get_result())

            # for result in result_str:
            #   print(result)

            export_results(results, training_size, threshold)
            export_results_str(result_str, training_size, threshold)

    end_time = datetime.now()
    print(f"Starting time: {start_time}")
    print(f"Ending time: {end_time}")
    print(f"Time difference: {end_time - start_time}")
