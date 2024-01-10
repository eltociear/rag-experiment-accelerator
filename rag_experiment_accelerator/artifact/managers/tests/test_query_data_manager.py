from rag_experiment_accelerator.artifact.managers.query_data_manager import (
    QueryDataManager,
)


def test_get_output_filename():
    index_name = "index_name"
    query_output_manager = QueryDataManager("")
    output_filename = query_output_manager.get_output_filename(index_name)
    assert output_filename == f"eval_output_{index_name}.jsonl"


def test_get_output_filepath():
    output_dir = "output"
    index_name = "index_name"
    manager = QueryDataManager(output_dir)
    output_filename = manager.get_output_filepath(index_name)
    assert output_filename == f"{output_dir}/{manager.get_output_filename(index_name)}"
