import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_building"))
from common import FEATURES, TARGET  # noqa: E402
from prep import clean  # noqa: E402


def _row(**kw):
    base = dict(CustomerID=1, ProdTaken=1, Age=30.0, TypeofContact="Self Enquiry", CityTier=1,
                DurationOfPitch=10.0, Occupation="Salaried", Gender="Male", NumberOfPersonVisiting=2,
                NumberOfFollowups=3.0, ProductPitched="Basic", PreferredPropertyStar=3.0,
                MaritalStatus="Single", NumberOfTrips=2.0, Passport=1, PitchSatisfactionScore=3,
                OwnCar=0, NumberOfChildrenVisiting=0.0, Designation="Executive", MonthlyIncome=20000.0)
    base.update(kw)
    return base


def test_clean_fixes_gender_caps_outliers_and_drops_columns():
    df = pd.DataFrame([_row(Gender="Fe Male", DurationOfPitch=127.0, NumberOfTrips=22.0),
                       _row(CustomerID=2), _row(CustomerID=3)])   # rows 2 and 3 duplicate once the ID is gone
    out = clean(df)
    assert list(out.columns) == FEATURES + [TARGET]
    assert "Fe Male" not in set(out["Gender"])
    assert out["DurationOfPitch"].max() <= 60 and out["NumberOfTrips"].max() <= 12
    assert len(out) == 2
