from typing import  Any, Callable, List, Optional, Protocol, Tuple
from langchain_core.documents import Document

class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: List[Document], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""



def split_list_of_docs(
    docs: List[Document], length_func: Callable, token_max: int, **kwargs: Any
) -> List[List[Document]]:
    """Split Documents into subsets that each meet a cumulative length constraint.

    Args:
        docs: The full list of Documents.
        length_func: Function for computing the cumulative length of a set of Documents.
        token_max: The maximum cumulative length of any subset of Documents.
        **kwargs: Arbitrary additional keyword params to pass to each call of the
            length_func.

    Returns:
        A List[List[Document]].
    """
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError(
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list



def collapse_docs(
    docs: List[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    """Execute a collapse function on a set of documents and merge their metadatas.

    Args:
        docs: A list of Documents to combine.
        combine_document_func: A function that takes in a list of Documents and
            optionally addition keyword parameters and combines them into a single
            string.
        **kwargs: Arbitrary additional keyword params to pass to the
            combine_document_func.

    Returns:
        A single Document with the output of combine_document_func for the page content
            and the combined metadata's of all the input documents. All metadata values
            are strings, and where there are overlapping keys across documents the
            values are joined by ", ".
    """
    result = combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)