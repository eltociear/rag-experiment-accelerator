from rag_experiment_accelerator.artifact.models.qa_data import QAData


def test_to_dict():
    qa = QAData("user_prompt", "output_prompt", "context")
    qa_dict = qa.to_dict()
    assert qa_dict["user_prompt"] == qa.user_prompt
    assert qa_dict["output_prompt"] == qa.output_prompt
    assert qa_dict["context"] == qa.context
