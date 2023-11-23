from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parents[1] / "data"


def get_features_and_target_from_csv(path: Path):
    df = pd.read_csv(path)
    features = df.drop(columns=['target']).astype(float)
    target = df['target'].astype(bool)
    return features, target
