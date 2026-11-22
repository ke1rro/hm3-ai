from pathlib import Path

import numpy as np
from scipy import sparse

from hm3.baseline import BaselineCandidate, build_pipeline


def test_bounded_refit_preserves_sklearn_vocabulary_idf_and_predictions(tmp_path: Path) -> None:
    """Catch changing the scientific TF-IDF recipe while reducing fit peak memory."""
    from hm3.baseline_export import fit_bounded_final_pipeline

    texts = [
        f"{topic} {topic} shared shared alpha beta gamma repeated {index % 4} Äpfel groß\n\nEND"
        for index in range(36)
        for topic in ["Human", "Machine", "Mixed"]
    ]
    labels = [label for _ in range(36) for label in ["human", "ai", "fusion"]]
    candidate = BaselineCandidate(
        "word_char_tfidf_logistic_regression_c1", "word_char", "logistic_regression", 1.0
    )
    settings = {"word_max_features": 12, "char_max_features": 25, "min_df": 3}
    expected = build_pipeline(candidate, **settings).fit(texts, labels)
    actual = fit_bounded_final_pipeline(texts, labels, candidate, tmp_path, "fixture", **settings)
    for (_, old), (_, new) in zip(
        expected.named_steps["features"].transformer_list,
        actual.named_steps["features"].transformer_list,
        strict=True,
    ):
        assert old.vocabulary_ == new.vocabulary_
        assert np.allclose(old.idf_, new.idf_, atol=1e-7)
    expected_features = expected.named_steps["features"].transform(texts)
    actual_features = actual.named_steps["features"].transform(texts)
    assert sparse.issparse(actual_features)
    assert np.allclose(expected_features.toarray(), actual_features.toarray(), atol=1e-7)
    assert np.array_equal(expected.predict(texts), actual.predict(texts))
    assert np.allclose(
        expected.named_steps["classifier"].coef_, actual.named_steps["classifier"].coef_, atol=1e-5
    )
    restored = fit_bounded_final_pipeline(texts, labels, candidate, tmp_path, "fixture", **settings)
    assert np.array_equal(restored.predict(texts), actual.predict(texts))
