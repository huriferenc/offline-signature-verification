#!/usr/bin/env python
import altair as alt
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from classifier.classifier import CLF_TYPE

from config import (
    CLASSIFIER_RESULT_FNAME_2,
    CLASSIFIER_RESULTS_FOLDER_FORMAT,
    ERROR_RATES_FNAME,
    ERROR_RATES_FOLDER,
    MAX_L,
    MIN_L,
)

MIN_THRESHOLD = MIN_L
MAX_THRESHOLD = 9

# CLASSIFIERS: list[CLF_TYPE] = ["linear", "rbf", "poly", "sigmoid", "random_forest"]
CLASSIFIERS: list[CLF_TYPE] = ["linear", "poly", "sigmoid", "random_forest"]

PERSONS = [
    {
        "id": 1,
        "time_index": 1715964571351,
    },
    {
        "id": 2,
        "time_index": 1715977512958,
    },
    {
        "id": 6,
        "time_index": 1716067236732,
    },
    {
        "id": 14,
        "time_index": 1716024937613,
    },
    {
        "id": 15,
        "time_index": 1716069105771,
    },
    {
        "id": 19,
        "time_index": 1716037988290,
    },
]

TRAINING_SIZES = [24, 48]  # 24 48

USE_COLS = ["classifier", "n", "p", "tn", "fp", "fn", "tp"]

NOW_TIME_INDEX = round(datetime.now().timestamp() * 1000)


def get_error_rate_dataframes(
    person_id: int, time_index: int, training_size: int, classifier: CLF_TYPE
):
    FRR = np.array([])
    FAR = np.array([])
    AER = np.array([])

    for threshold in range(MIN_THRESHOLD, MAX_THRESHOLD + 1):
        df = pd.read_csv(
            Path.cwd()
            / CLASSIFIER_RESULTS_FOLDER_FORMAT.format(
                person_id=person_id, time_index=time_index
            )
            / CLASSIFIER_RESULT_FNAME_2.format(
                person_id=person_id,
                training_size=training_size,
                l=threshold,
                time_index=time_index,
            ),
            usecols=USE_COLS,
        )

        error_values = df[df["classifier"] == classifier][
            ["n", "p", "tn", "fp", "fn", "tp"]
        ]

        n = error_values["n"]
        p = error_values["p"]
        tn = error_values["tn"]
        fp = error_values["fp"]
        fn = error_values["fn"]
        tp = error_values["tp"]

        frr = fn / n
        far = fp / p
        aer = (frr + far) / 2

        FRR = np.append(FRR, frr)
        FAR = np.append(FAR, far)
        AER = np.append(AER, aer)

    return [FRR, FAR, AER]


def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ["c1", "c2", "values"]
    df["DF"] = name
    return df


def export_error_rates(
    training_size, classifier, frr_error_rates, far_error_rates, aer_error_rates
):
    person_ids = [str(person["id"]) for person in PERSONS]
    person_ids_str = ",".join(person_ids)

    df_frr = pd.DataFrame(
        frr_error_rates.transpose(),
        index=[f"Threshold {i}" for i in range(MIN_THRESHOLD, MAX_THRESHOLD + 1)],
        columns=[f"Person-{id}" for id in person_ids],
    )
    df_far = pd.DataFrame(
        far_error_rates.transpose(),
        index=[f"Threshold {i}" for i in range(MIN_THRESHOLD, MAX_THRESHOLD + 1)],
        columns=[f"Person-{id}" for id in person_ids],
    )
    df_aer = pd.DataFrame(
        aer_error_rates.transpose(),
        index=[f"Threshold {i}" for i in range(MIN_THRESHOLD, MAX_THRESHOLD + 1)],
        columns=[f"Person-{id}" for id in person_ids],
    )

    df_frr = prep_df(df_frr, "FRR")
    df_far = prep_df(df_far, "FAR")
    df_aer = prep_df(df_aer, "AER")

    df = pd.concat([df_frr, df_far, df_aer])

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("c2:N", title=None),
            y=alt.Y(
                "sum(values):Q",
                axis=alt.Axis(grid=False, title=None),
                sort=None,
            ),
            column=alt.Column("c1:N", title=None, sort=None),
            color=alt.Color(
                "DF:N", scale=alt.Scale(range=["#2CA02C", "#FF7F0E", "#1F77B4"])
            ),
        )
        .configure_view(strokeOpacity=0)
    )

    (
        Path.cwd()
        / (
            ERROR_RATES_FOLDER.format(
                person_ids=person_ids_str, time_index=NOW_TIME_INDEX
            )
        )
    ).mkdir(parents=True, exist_ok=True)

    chart.save(
        Path.cwd()
        / (
            ERROR_RATES_FOLDER.format(
                person_ids=person_ids_str, time_index=NOW_TIME_INDEX
            )
        )
        / ERROR_RATES_FNAME.format(
            person_ids=person_ids_str,
            classifier=classifier,
            training_size=training_size,
            time_index=NOW_TIME_INDEX,
        ),
        ppi=600,
    )


if __name__ == "__main__":
    for training_size in TRAINING_SIZES:
        for classifier in CLASSIFIERS:
            frr_error_rates = np.zeros(
                [len(PERSONS), MAX_THRESHOLD - MIN_THRESHOLD + 1], dtype=np.float16
            )
            far_error_rates = np.zeros(
                [len(PERSONS), MAX_THRESHOLD - MIN_THRESHOLD + 1], dtype=np.float16
            )
            aer_error_rates = np.zeros(
                [len(PERSONS), MAX_THRESHOLD - MIN_THRESHOLD + 1], dtype=np.float16
            )

            index = 0
            for person in PERSONS:
                person_id = person["id"]
                time_index = person["time_index"]

                FRR, FAR, AER = get_error_rate_dataframes(
                    person_id, time_index, training_size, classifier
                )

                frr_error_rates[index] = FRR
                far_error_rates[index] = FAR
                aer_error_rates[index] = AER

                index = index + 1

            export_error_rates(
                training_size,
                classifier,
                frr_error_rates,
                far_error_rates,
                aer_error_rates,
            )
