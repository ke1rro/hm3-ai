import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from resource import RUSAGE_SELF, getrusage

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from hm3.baseline import BaselineCandidate, build_pipeline
from hm3.validation import fingerprint_file

logger = logging.getLogger(__name__)


def fit_streaming_vectorizer(vectorizer: TfidfVectorizer, texts: list[str]) -> TfidfVectorizer:
    """Fit the baseline vocabulary and IDF without allocating its unpruned corpus matrix.

    Count term and document frequencies using the configured sklearn analyzer, then
    apply sklearn's alphabetical ordering and frequency-based vocabulary selection.
    Initialize the fixed vocabulary on one training document and replace its IDF
    with the full-training document-frequency calculation in the configured dtype.

    Args:
        vectorizer: Baseline word or character vectorizer to fit in place.
        texts: Complete authorized training texts in their original order.

    Returns:
        Fitted vectorizer with the original settings and bounded vocabulary.
    """
    analyzer = vectorizer.build_analyzer()
    term_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    for index, text in enumerate(texts, start=1):
        counts = Counter(analyzer(text))
        term_counts.update(counts)
        document_counts.update(counts.keys())
        if index % 10000 == 0:
            peak = getrusage(RUSAGE_SELF).ru_maxrss / (
                1024**3 if sys.platform == "darwin" else 1024**2
            )
            logger.info(
                "%s frequency pass: %s/%s documents; peak RSS %.2f GiB",
                vectorizer.analyzer,
                index,
                len(texts),
                peak,
            )
    terms = sorted(term_counts)
    frequencies = np.fromiter((term_counts[term] for term in terms), dtype=np.int64)
    documents = np.fromiter((document_counts[term] for term in terms), dtype=np.int64)
    minimum = (
        vectorizer.min_df if isinstance(vectorizer.min_df, int) else vectorizer.min_df * len(texts)
    )
    maximum = (
        vectorizer.max_df if isinstance(vectorizer.max_df, int) else vectorizer.max_df * len(texts)
    )
    mask = (documents >= minimum) & (documents <= maximum)
    if vectorizer.max_features is not None and mask.sum() > vectorizer.max_features:
        candidates = np.flatnonzero(mask)
        selected = candidates[(-frequencies[mask]).argsort()[: vectorizer.max_features]]
        mask[:] = False
        mask[selected] = True
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError("No terms remain after the baseline document-frequency limits")
    vocabulary = {terms[index]: position for position, index in enumerate(indices)}
    vectorizer.set_params(vocabulary=vocabulary)
    vectorizer.fit(texts[:1])
    document_frequency = documents[indices].astype(vectorizer.dtype)
    document_frequency += int(vectorizer.smooth_idf)
    inverse_frequency = np.full_like(document_frequency, len(texts) + int(vectorizer.smooth_idf))
    np.divide(inverse_frequency, document_frequency, out=inverse_frequency)
    np.log(inverse_frequency, out=inverse_frequency)
    inverse_frequency += 1
    vectorizer.idf_ = inverse_frequency
    logger.info("Fitted %s vocabulary: %s features", vectorizer.analyzer, len(vocabulary))
    return vectorizer


def _load_checkpoint(path: Path, identity: dict) -> bool:
    record_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() and not record_path.exists():
        return False
    if not path.exists() or not record_path.exists():
        raise ValueError(f"Incomplete bounded-export checkpoint: {path}")
    record = json.loads(record_path.read_text())
    if record["identity"] != identity or record["sha256"] != fingerprint_file(path):
        raise ValueError(f"Incompatible or corrupted bounded-export checkpoint: {path}")
    return True


def _publish(path: Path, identity: dict) -> None:
    record = {"identity": identity, "sha256": fingerprint_file(path)}
    destination = path.with_suffix(path.suffix + ".json")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(destination)


def fit_bounded_final_pipeline(
    texts: list[str],
    labels: list[str],
    candidate: BaselineCandidate,
    directory: Path,
    source_identity: str,
    word_max_features: int = 40000,
    char_max_features: int = 60000,
    min_df: int = 3,
) -> Pipeline:
    """Fit or restore the combined final baseline with staged bounded-vocabulary features.

    Persist fitted vectorizers, the final sparse training matrix and the classifier
    as separate verified stages. Never build an unbounded word/character count
    matrix. Keep all scientific settings and the full training population unchanged.

    Args:
        texts: All labeled training documents, with no validation or test fitting.
        labels: Original labels aligned with texts.
        candidate: Selected combined word/character baseline recipe.
        directory: Private stage checkpoint directory.
        source_identity: Fingerprint of the unchanged labeled CSV.
        word_max_features: Original word vocabulary bound.
        char_max_features: Original character vocabulary bound.
        min_df: Original document-frequency threshold.

    Returns:
        Fitted sklearn pipeline usable for batched test prediction.
    """
    if candidate.representation != "word_char":
        raise ValueError("This export path supports the selected combined baseline")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "source_sha256": source_identity,
        "candidate": asdict(candidate),
        "rows": len(texts),
        "word_max_features": word_max_features,
        "char_max_features": char_max_features,
        "min_df": min_df,
        "implementation_sha256": fingerprint_file(Path(__file__)),
        "label_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
    }
    model_path = directory / "final_pipeline.joblib"
    if _load_checkpoint(model_path, identity):
        logger.info("Restored completed final classifier")
        return joblib.load(model_path)
    pipeline = build_pipeline(candidate, word_max_features, char_max_features, min_df)
    feature_path = directory / "fitted_vectorizers.joblib"
    if _load_checkpoint(feature_path, identity):
        features = joblib.load(feature_path)
        pipeline.set_params(features=features)
        logger.info("Restored fitted full-training vectorizers")
    else:
        features = pipeline.named_steps["features"]
        for _, vectorizer in features.transformer_list:
            fit_streaming_vectorizer(vectorizer, texts)
        temporary = feature_path.with_suffix(".joblib.tmp")
        joblib.dump(features, temporary, compress=3)
        temporary.replace(feature_path)
        _publish(feature_path, identity)
    matrix_path = directory / "training_features.npz"
    if _load_checkpoint(matrix_path, identity):
        training_features = sparse.load_npz(matrix_path)
        logger.info("Restored full-training sparse feature matrix")
    else:
        chunks = []
        for start in range(0, len(texts), 512):
            chunks.append(features.transform(texts[start : start + 512]))
            if start % 5120 == 0:
                logger.info(
                    "Bounded feature transform: %s/%s documents",
                    min(start + 512, len(texts)),
                    len(texts),
                )
        training_features = sparse.vstack(chunks, format="csr")
        del chunks
        temporary_matrix = matrix_path.with_name("training_features.tmp.npz")
        sparse.save_npz(temporary_matrix, training_features, compressed=False)
        temporary_matrix.replace(matrix_path)
        _publish(matrix_path, identity)
    peak = getrusage(RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)
    logger.info(
        "Fitting selected classifier: shape=%s, nnz=%s, peak RSS=%.2f GiB",
        training_features.shape,
        training_features.nnz,
        peak,
    )
    pipeline.named_steps["classifier"].fit(training_features, labels)
    del training_features
    temporary = model_path.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, temporary, compress=3)
    temporary.replace(model_path)
    _publish(model_path, identity)
    logger.info("Published completed final classifier: %s", model_path)
    return pipeline
