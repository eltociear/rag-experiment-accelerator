from typing import Self
from rag_experiment_accelerator.artifact.models.artifact import Artifact


class QAData(Artifact):
    def __init__(self, user_prompt: str, output_prompt: str, context: str = ""):
        self.user_prompt = user_prompt
        self.output_prompt = output_prompt
        self.context = context
