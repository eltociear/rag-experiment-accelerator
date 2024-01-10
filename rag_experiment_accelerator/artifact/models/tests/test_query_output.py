from rag_experiment_accelerator.artifact.models.query_data import QueryOutput


def test_to_dict():
    output = QueryOutput(
        rerank="rerank",
        rerank_type="rerank_type",
        crossencoder_model="crossencoder_model",
        llm_re_rank_threshold="llm_re_rank_threshold",
        retrieve_num_of_documents="retrieve_num_of_documents",
        cross_encoder_at_k="cross_encoder_at_k",
        question_count="question_count",
        actual="actual",
        expected="expected",
        search_type="search_type",
        search_evals="search_evals",
        context="context",
    )
    output_dict = output.to_dict()
    assert output_dict["rerank"] == output.rerank
    assert output_dict["rerank_type"] == output.rerank_type
    assert output_dict["crossencoder_model"] == output.crossencoder_model
    assert output_dict["llm_re_rank_threshold"] == output.llm_re_rank_threshold
    assert output_dict["retrieve_num_of_documents"] == output.retrieve_num_of_documents
    assert output_dict["cross_encoder_at_k"] == output.cross_encoder_at_k
    assert output_dict["question_count"] == output.question_count
    assert output_dict["actual"] == output.actual
    assert output_dict["expected"] == output.expected
    assert output_dict["search_type"] == output.search_type
    assert output_dict["search_evals"] == output.search_evals
    assert output_dict["context"] == output.context


def test_from_dict():
    output_dict = {
        "rerank": "rerank",
        "rerank_type": "rerank_type",
        "crossencoder_model": "crossencoder_model",
        "llm_re_rank_threshold": "llm_re_rank_threshold",
        "retrieve_num_of_documents": "retrieve_num_of_documents",
        "cross_encoder_at_k": "cross_encoder_at_k",
        "question_count": "question_count",
        "actual": "actual",
        "expected": "expected",
        "search_type": "search_type",
        "search_evals": "search_evals",
        "context": "context",
    }
    output = QueryOutput.from_dict(output_dict)
    assert output.rerank == output_dict["rerank"]
    assert output.rerank_type == output_dict["rerank_type"]
    assert output.crossencoder_model == output_dict["crossencoder_model"]
    assert output.llm_re_rank_threshold == output_dict["llm_re_rank_threshold"]
    assert output.retrieve_num_of_documents == output_dict["retrieve_num_of_documents"]
    assert output.cross_encoder_at_k == output_dict["cross_encoder_at_k"]
    assert output.question_count == output_dict["question_count"]
    assert output.actual == output_dict["actual"]
    assert output.expected == output_dict["expected"]
    assert output.search_type == output_dict["search_type"]
    assert output.search_evals == output_dict["search_evals"]
    assert output.context == output_dict["context"]
