import faiss


def get_faiss_indexer(shape):
    indexer = faiss.IndexFlatL2(shape)

    return indexer
