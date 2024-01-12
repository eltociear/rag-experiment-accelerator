from typing import TypeVar

from rag_experiment_accelerator.file_handlers.writers.writer import Writer


V = TypeVar("V", bound=Writer)
