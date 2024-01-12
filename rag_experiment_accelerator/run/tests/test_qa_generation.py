from unittest.mock import patch, MagicMock
from pandas import DataFrame
from rag_experiment_accelerator.run.qa_generation import run

# @patch('rag_experiment_accelerator.run.qa_generation.get_default_az_cred')
# @patch('rag_experiment_accelerator.run.qa_generation.load_documents')
# @patch('rag_experiment_accelerator.run.qa_generation.Config')
# @patch('os.makedirs')
# def test_run(mock_makedirs, mock_config, mock_load_documents, mock_get_default_az_cred):
#     # Arrange
#     mock_config.return_value = Config('test_dir')
#     mock_get_default_az_cred.return_value = 'test_cred'
#     mock_load_documents.return_value = 'test_docs'
#     mock_makedirs.side_effect = None

#     # Act
#     run('test_dir')

#     # Assert
#     mock_makedirs.assert_called_once_with('test_dir/artifacts_dir', exist_ok=True)
#     mock_config.assert_called_once_with('test_dir')
#     mock_get_default_az_cred.assert_called_once()
#     mock_load_documents.assert_called_once_with('test_dir/DATA_FORMATS', 'test_dir/data_dir', 2000, 0)


@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
def test_run_success(
    mock_config,
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
    mock_generate_qna,
    mock_to_json,
    mock_create_data_asset,
):
    # Arrange
    mock_config.return_value = MagicMock()
    mock_get_default_az_cred.return_value = "test_cred"
    mock_load_documents.return_value = MagicMock()
    mock_makedirs.side_effect = None
    mock_df = MagicMock(DataFrame)
    mock_generate_qna.return_value = mock_df
    mock_create_data_asset.side_effect = None

    # Act
    run("test_dir")

    # Assert
    mock_makedirs.assert_called_once()
    mock_config.assert_called_once_with("test_dir")
    mock_get_default_az_cred.assert_called_once()
    mock_load_documents.assert_called_once()
    mock_generate_qna.assert_called_once()
    mock_df.to_json.assert_called_once()
    mock_create_data_asset.assert_called_once()


@patch("rag_experiment_accelerator.run.qa_generation.create_data_asset")
@patch("pandas.DataFrame.to_json")
@patch("rag_experiment_accelerator.run.qa_generation.generate_qna")
@patch("os.makedirs")
@patch("rag_experiment_accelerator.run.qa_generation.load_documents")
@patch("rag_experiment_accelerator.run.qa_generation.get_default_az_cred")
@patch("rag_experiment_accelerator.run.qa_generation.Config")
def test_run_makedirs_exception(
    mock_config,
    mock_get_default_az_cred,
    mock_load_documents,
    mock_makedirs,
    mock_generate_qna,
    mock_to_json,
    mock_create_data_asset,
):
    # Arrange
    mock_config.return_value = MagicMock()
    mock_get_default_az_cred.return_value = "test_cred"
    mock_load_documents.return_value = MagicMock()
    mock_makedirs.side_effect = Exception("test exception")

    # # Act and Assert
    # with assertRaises(Exception) as context:
    #     run('test_dir')
    # assertTrue('Unable to create the' in str(context.exception))
