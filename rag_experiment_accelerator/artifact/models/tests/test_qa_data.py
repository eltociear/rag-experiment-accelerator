from rag_experiment_accelerator.artifact.models.qa_data import QAData


def test_to_dict():
    qa = QAData("user_prompt", "output_prompt", "context")
    qa_dict = qa.to_dict()
    assert qa_dict["user_prompt"] == qa.user_prompt
    assert qa_dict["output_prompt"] == qa.output_prompt
    assert qa_dict["context"] == qa.context


def test_from_dict():
    qa_dict = {
        "user_prompt": "user_prompt",
        "output_prompt": "output_prompt",
        "context": "context",
    }
    qa = QAData.from_dict(qa_dict)
    assert qa.user_prompt == qa_dict["user_prompt"]
    assert qa.output_prompt == qa_dict["output_prompt"]
    assert qa.context == qa_dict["context"]
