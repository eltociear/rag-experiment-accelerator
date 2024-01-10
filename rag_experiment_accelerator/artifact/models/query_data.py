from rag_experiment_accelerator.artifact.models.artifact import Artifact


class QueryData(Artifact):
    def __init__(
        self,
        rerank: bool,
        rerank_type: str,
        crossencoder_model: str,
        llm_re_rank_threshold: int,
        retrieve_num_of_documents: int,
        cross_encoder_at_k: int,
        question_count: int,
        actual: str,
        expected: str,
        search_type: str,
        search_evals: list,
        context: str,
    ):
        self.rerank = rerank
        self.rerank_type = rerank_type
        self.crossencoder_model = crossencoder_model
        self.llm_re_rank_threshold = llm_re_rank_threshold
        self.retrieve_num_of_documents = retrieve_num_of_documents
        self.cross_encoder_at_k = cross_encoder_at_k
        self.question_count = question_count
        self.actual = actual
        self.expected = expected
        self.search_type = search_type
        self.search_evals = search_evals
        self.context = context

    # @classmethod
    # def from_dict(cls, data: dict) -> Self:
    #     return cls(
    #         rerank=data["rerank"],
    #         rerank_type=data["rerank_type"],
    #         crossencoder_model=data["crossencoder_model"],
    #         llm_re_rank_threshold=data["llm_re_rank_threshold"],
    #         retrieve_num_of_documents=data["retrieve_num_of_documents"],
    #         cross_encoder_at_k=data["cross_encoder_at_k"],
    #         question_count=data["question_count"],
    #         actual=data["actual"],
    #         expected=data["expected"],
    #         search_type=data["search_type"],
    #         search_evals=data["search_evals"],
    #         context=data["context"],
    #     )
