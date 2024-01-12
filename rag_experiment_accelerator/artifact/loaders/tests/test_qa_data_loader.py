from rag_experiment_accelerator.artifact.loaders.qa_data_loader import QADataLoader
from rag_experiment_accelerator.artifact.loaders.tests.fixtures import temp_dir
from rag_experiment_accelerator.artifact.models.qa_data import QAData
from rag_experiment_accelerator.artifact.writers.qa_data_writer import QADataWriter


def test_loads(temp_dir: str):
    test_data = [
        QAData("user_prompt1", "output_prompt1", "context1"),
        QAData("user_prompt2", "output_prompt2", "context2"),
    ]
    filename = "test.jsonl"
    path = f"{temp_dir}/{filename}"
    writer = QADataWriter(path)

    # save the data to a temp file so it can be loaded
    writer.save_all(test_data)

    # load the data
    loader = QADataLoader(writer.get_write_path())
    loaded_data = loader.load_all()

    # assertions
    for i, d in enumerate(loaded_data):
        assert d.user_prompt == test_data[i].user_prompt
        assert d.output_prompt == test_data[i].output_prompt
        assert d.context == test_data[i].context
