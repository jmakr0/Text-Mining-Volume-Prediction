import numpy as np


class TfIdf:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_tf_idf(self, document_sequences):
        tf_idf_matrix = []
        for document_sequence in document_sequences:
            # number of docs a word is in
            docs_vector = np.array(
                [self.tokenizer.index_docs[index] if index else len(document_sequences) for index in document_sequence])
            idf_vector = np.log(len(document_sequences) / docs_vector)

            indices, counts = np.unique(np.array([i for i in document_sequence if i]), return_counts=True)
            index_counts = dict(zip(indices, counts))

            tf_vector = np.array(
                [(index_counts[index] if index else 0) / (counts.max() if len(counts) else 1) for index in
                 document_sequence])

            tf_idf_matrix.append(tf_vector * idf_vector)

        return tf_idf_matrix
