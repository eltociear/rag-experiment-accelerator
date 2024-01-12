import json
import os
import pathlib

from rag_experiment_accelerator.artifact.models.index_data import IndexData
from rag_experiment_accelerator.artifact.writers.index_data_writer import (
    IndexDataWriter,
)
from rag_experiment_accelerator.artifact.writers.tests.fixtures import temp_dirname


def test_save(temp_dirname: str):
    writer = IndexDataWriter(temp_dirname)
    path = writer.output_filepath
    indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]
    # save the data
    writer.save_all(indexes)

    # check that the data was saved
    assert pathlib.Path(path).exists()
    with open(path, "r") as file:
        content = file.read()
        data = json.loads(content)
        assert [i for i in data] == [i.name for i in indexes]


def test_should_archive_false(temp_dirname: str):
    prev_indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]

    new_indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]
    # save the data
    writer = IndexDataWriter(temp_dirname)
    writer.save_all(prev_indexes)

    should_archive = writer._should_archive(new_indexes, prev_indexes)

    assert should_archive is False


def test_should_archive_true(temp_dirname: str):
    prev_indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]

    new_indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="different_embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]
    # save the data
    writer = IndexDataWriter(temp_dirname)
    writer.save_all(prev_indexes)

    should_archive = writer._should_archive(new_indexes, prev_indexes)

    assert should_archive is True


def test_handle_archive(temp_dirname: str):
    writer = IndexDataWriter(temp_dirname)
    path = writer.output_filepath
    indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]
    # save the data
    writer.save_all(indexes)

    indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]

    # archive the file
    archive_filepath = writer.handle_archive(new_indexes=[], prev_indexes=indexes)

    # assertions
    assert pathlib.Path(f"{writer.directory}/archive").exists()
    assert pathlib.Path(archive_filepath).exists()
    assert not pathlib.Path(path).exists()


def test_handle_archive_no_op(temp_dirname: str):
    writer = IndexDataWriter(temp_dirname)
    path = writer.output_filepath
    indexes = [
        IndexData(
            prefix="prefix1",
            chunk_size=1,
            overlap=1,
            embedding_model_name="embedding_model_name1",
            ef_construction=1,
            ef_search=1,
        ),
        IndexData(
            prefix="prefix2",
            chunk_size=2,
            overlap=2,
            embedding_model_name="embedding_model_name2",
            ef_construction=2,
            ef_search=2,
        ),
    ]
    # save the data
    writer.save_all(indexes)

    # archive the file
    archive_filepath = writer.handle_archive(new_indexes=indexes, prev_indexes=indexes)

    # assertions
    files = os.listdir(writer.archive_dir)
    assert len(files) == 0
    assert archive_filepath is None
    assert pathlib.Path(path).exists()
