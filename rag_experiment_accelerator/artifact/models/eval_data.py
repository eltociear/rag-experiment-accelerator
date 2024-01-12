from typing import Self
from rag_experiment_accelerator.artifact.models.artifact import Artifact
import pandas as pd


class EvalData(pd.DataFrame, Artifact):
    pass
