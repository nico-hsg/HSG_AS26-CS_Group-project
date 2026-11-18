"""Schnelle Offline-Tests: python -m unittest test_snowproof_ml.py"""
import unittest

import numpy as np
import pandas as pd
from snowproof_ml import label_good_weeks, split_dataset


class TestSnowproofML(unittest.TestCase):
    @staticmethod
    def example(values, start="2020-02-01"):
        return pd.DataFrame({
            "station_code": ["DAV"] * len(values),
            "date": pd.date_range(start, periods=len(values), freq="D"),
            "snow_depth_cm": values,
        })

    def test_good_week(self):
        result = label_good_weeks(self.example([50] * 7))
        self.assertEqual(result.loc[0, "good_week"], 1)
        self.assertEqual(result.loc[0, "average_snow_depth_cm"], 50)

    def test_bad_week(self):
        result = label_good_weeks(self.example([49] * 7))
        self.assertEqual(result.loc[0, "good_week"], 0)

    def test_missing_measurement(self):
        result = label_good_weeks(self.example([60, 60, np.nan, 60, 60, 60, 60]))
        self.assertTrue(pd.isna(result.loc[0, "good_week"]))

    def test_missing_calendar_day(self):
        data = self.example([60] * 8)
        data = data[data["date"] != pd.Timestamp("2020-02-04")]
        self.assertTrue(label_good_weeks(data).empty)

    def test_no_overlap_between_train_and_val(self):
        data = pd.DataFrame({
            "week_start": pd.to_datetime(["2021-12-25", "2021-12-28", "2022-01-01"]),
            "week_end": pd.to_datetime(["2021-12-31", "2022-01-03", "2022-01-07"]),
        })
        train, val = split_dataset(data)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)


if __name__ == "__main__":
    unittest.main()
