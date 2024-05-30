import os
import argparse
import json
from collections import Counter, defaultdict
import string
import nltk
import tqdm
import numpy as np
import scipy.sparse as sp
import time
import bottleneck as bn
import numba as nb
from typing import List


@nb.jit(nopython=True, cache=True, fastmath=True)
def compute_token_scores(
    indices, doclens, freq, b, k1, idf, avg_document_length, scores
):
    one = np.float32(1)

    for i in range(len(indices)):
        freq_val = freq[i]
        doclen_val = doclens[indices[i]] / avg_document_length
        scores[i] = (
            idf * (freq_val * (k1 + one)) / (freq_val + k1 * (one - b + b * doclen_val))
        )


@nb.njit(cache=True, fastmath=True)
def precompute_token_scores(row, indices, numerator, denominator_term):
    denominator = denominator_term[indices]
    np.add(row, denominator, denominator)
    np.divide(row, denominator, denominator)
    denominator *= numerator
    return denominator


class Bm25Index:
    nltk.download("stopwords")
    stopwords = set(nltk.corpus.stopwords.words("english"))

    def __init__(
        self,
        collection_path,
        index_root,
        index_name,
        precompute_scores: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):

        if collection_path:
            self.collection = Bm25Index.load_collection(collection_path)

        index_path = os.path.join(index_root, index_name)
        self.load_index(index_path)

        self.k1 = np.float32(k1)
        self.b = np.float32(b)

        self.precompute_scores = precompute_scores
        if self.precompute_scores:
            self.scores = []

            numerator_term = np.float32(self.k1 + 1)
            denominator_term = self.k1 * (
                1 - self.b + self.b * (self.doclens / self.avg_document_length)
            ).astype(np.float32)

            print(f"Precomputing scores for {self.index.shape[0]} tokens...")
            for i in tqdm.tqdm(range(self.index.shape[0])):
                row = self.index.data[self.index.indptr[i] : self.index.indptr[i + 1]]

                token_indices = self.indices[i]

                df = row.shape[0]

                idf = np.log(
                    (self.num_documents - df + 0.5) / (df + 0.5) + 1.0, dtype=np.float32
                )
                token_scores = precompute_token_scores(
                    row, token_indices, idf * numerator_term, denominator_term
                )
                self.scores.append(token_scores)

    def load_index(self, index_path):
        print("Loading index...")
        self.index = sp.load_npz(os.path.join(index_path, "index.npz"))

        print("Converting index to csr format...")
        self.index = self.index.tocsr()
        self.indices = np.split(self.index.indices, self.index.indptr[1:-1])
        self.index.data.flags.writeable = False

        print("Loading vocabulary...")
        self.vocabulary = []
        self.inverted_vocabulary = {}
        with open(os.path.join(index_path, "vocabulary.txt"), "r") as f:
            i = 0
            for line in f:
                term = line.strip()
                self.vocabulary.append(term)
                self.inverted_vocabulary[term] = i
                i += 1

        print("Loading metadata...")
        with open(os.path.join(index_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        self.pid_offset = metadata["pid_offset"]
        self.num_documents = metadata["num_documents"]
        self.avg_document_length = metadata["avg_document_length"]

        print("Loading doclens...")
        self.doclens = np.zeros(
            shape=(self.num_documents + self.pid_offset), dtype=np.uint16
        )
        with open(os.path.join(index_path, "doclens.tsv"), "r") as f:
            for line in f:
                pid, doclen = line.strip().split("\t")
                pid = int(pid)
                doclen = int(doclen)
                self.doclens[pid] = doclen

        print("Finished loading index.")

    def get_token_scores(self, token):
        if token not in self.inverted_vocabulary:
            return None

        token_idx = self.inverted_vocabulary[token]

        token_indices = self.indices[token_idx]

        if self.precompute_scores:
            token_data = self.scores[token_idx]
        else:
            row = self.index.data[
                self.index.indptr[token_idx] : self.index.indptr[token_idx + 1]
            ]

            df = row.shape[0]

            idf = np.log(
                (self.num_documents - df + 0.5) / (df + 0.5) + 1.0, dtype=np.float32
            )

            token_data = np.zeros(row.shape, dtype=np.float32)

            compute_token_scores(
                token_indices,
                self.doclens,
                row,
                self.b,
                self.k1,
                idf,
                self.avg_document_length,
                token_data,
            )

        token_scores = sp.csr_array(
            (token_data, token_indices, [0, token_data.shape[0]]),
            shape=(1, self.num_documents),
        )

        return token_scores

    def get_topk(self, scores, k=10):
        pids = scores.indices
        scores = scores.data

        if scores.shape[-1] > k:
            ind = bn.argpartition(scores, scores.size - k)[-k:]
            topk = (pids[ind].tolist(), scores[ind].tolist())
        else:
            topk = (pids.tolist(), scores.tolist())

        return topk

    def search_all(self, query_batch: List[str], k: int = 10, verbose: bool = False):
        rankings = []
        for query in query_batch:
            rankings.append(self.search(query, k, verbose))
        return rankings

    def search(
        self,
        query: str,
        k: int = 10,
        weight_by_frequency: bool = False,
        verbose: bool = False,
    ):
        tokens = Bm25Index.preprocess(query, Bm25Index.stopwords)
        if weight_by_frequency:
            raise NotImplementedError("Need to re-implement weight-by-frequency")
            # token_counts = Counter(tokens)

        tokens = [token for token in set(tokens) if token in self.inverted_vocabulary]

        if weight_by_frequency:
            multipliers = [token_counts[token] for token in tokens]
        else:
            multipliers = None

        token_scores = [self.get_token_scores(token) for token in tokens]
        if len(token_scores) > 0:
            token_scores.sort(key=lambda x: x.getnnz(), reverse=True)
            scores = reduce(sum, token_scores)
            topk = self.get_topk(scores, k=k)
        else:
            topk = ([], [])

        if verbose:
            print(f"Query: {query}\n")
            for i, (pid, score) in enumerate(list(zip(*topk))):
                print(f"{i+1}) Document {pid}: score = {score}")
                print("*" * 80)
                print(self.collection[pid])
                print("\n")

        return topk

    @classmethod
    def load_collection(cls, collection_path):
        docs = {}
        print(f"Loading collection from {collection_path}...")
        with open(collection_path, "r") as f:
            for line in tqdm.tqdm(f.readlines()[1:]):
                cols = line.strip().split("\t")
                pid = int(cols[0])
                doc = cols[1]
                docs[pid] = doc
        return docs

    # Normalizes and filters tokens
    @classmethod
    def preprocess(cls, text, stopwords):
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = [token for token in text.lower().split() if token not in stopwords]
        return tokens

    # Function to build the inverted index
    @classmethod
    def build_index(cls, collection_path, output_path, index_name):

        index_path = os.path.join(output_path, index_name)
        if not os.path.isdir(index_path):
            try:
                os.mkdir(index_path)
            except Exception as e:
                print(f"Failed to create index directory: {e}")
                return

        nltk.download("stopwords")
        stopwords = set(nltk.corpus.stopwords.words("english"))

        docs = Bm25Index.load_collection(collection_path)

        inverted_index = defaultdict(list)
        doclens = {}

        pids = sorted(list(docs.keys()))

        print(f"Building index...")
        for pid, doc in docs.items():
            tokens = Bm25Index.preprocess(doc, stopwords)
            doclens[pid] = len(tokens)

            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            for term, freq in term_freq.items():
                inverted_index[term].append((pid, freq))

        terms = list(inverted_index.keys())

        nterms = len(terms)
        npids = len(pids)
        avg_doclen = np.mean(list(doclens.values()))

        print(f"Converting index to sparse format - {nterms} terms and {npids} pids...")
        sparse_index = sp.dok_array((nterms, min(pids) + max(pids)), dtype=np.uint16)
        for i, term in tqdm.tqdm(enumerate(terms)):
            for (pid, freq) in inverted_index[term]:
                sparse_index[i, pid] = freq

        print(f"Saving sparse index...")
        sp.save_npz(os.path.join(index_path, f"index.npz"), sparse_index.tocoo())

        print("Saving vocabulary...")
        with open(os.path.join(index_path, "vocabulary.txt"), "w") as f:
            for term in terms:
                f.write(f"{term}\n")

        print("Saving doc lengths...")
        with open(os.path.join(index_path, "doclens.tsv"), "w") as f:
            for pid, doclen in doclens.items():
                f.write(f"{pid}\t{doclen}\n")

        print("Saving metadata...")
        metadata = {
            "pid_offset": min(pids),
            "num_documents": npids,
            "avg_document_length": avg_doclen,
        }
        with open(os.path.join(index_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        print("Finished building index.")


def main(args):
    collection_path = args.collection_path
    if args.verbose:
        assert collection_path
    start = time.time()
    index = Bm25Index(
        collection_path=collection_path,
        index_root=args.output_path,
        index_name=args.index_name,
        precompute_scores=args.precompute_scores,
    )
    end = time.time()
    print(f"Loaded index in {(end - start):.2f} seconds")

    queries = []
    with open(os.path.join(os.environ["DATA_PATH"], args.queries), "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries.append(query)

    print("Warming up...")
    for query in tqdm.tqdm(queries[:10]):
        index.search(query, k=10, verbose=False)

    print("Benchmarking...")
    query_batches = np.array_split(queries, len(queries) // args.batch_size)
    latencies = []
    if args.n is not None:
        query_batches = query_batches[: args.n]
    for query_batch in tqdm.tqdm(query_batches):
        start = time.time()
        if args.batch_size == 1:
            index.search(
                query_batch[0],
                k=args.k,
                weight_by_frequency=args.weight_by_frequency,
                verbose=args.verbose,
            )
        else:
            index.search_all(
                query_batch,
                k=args.k,
                weight_by_frequency=args.weight_by_frequency,
                verbose=args.verbose,
            )
        end = time.time()
        latencies.append(end - start)
    print(f"Average latency: {(np.mean(latencies) * 1e3):.2f} ms")
    print(f"Median latency: {(np.median(latencies) * 1e3):.2f} ms")
    print(f"P95 latency: {(np.percentile(latencies, 95) * 1e3):.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 indexer")
    parser.add_argument(
        "--collection_path", type=str, default=None, help="Path to collection"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--index_name", type=str, required=True, help="Index name")
    parser.add_argument("-k", type=int, default=1000, help="Number of docs to return")
    parser.add_argument("-n", type=int, default=None, help="Number of queries")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--precompute_scores",
        action="store_true",
        default=False,
        help="Precomputes scores (faster but takes more memory)",
    )
    parser.add_argument(
        "--weight_by_frequency",
        action="store_true",
        default=False,
        help="Weight query terms by frequency",
    )
    parser.add_argument(
        "--queries", type=str, required=True, help="Path to queries tsv file"
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose")
    args = parser.parse_args()
    main(args)
