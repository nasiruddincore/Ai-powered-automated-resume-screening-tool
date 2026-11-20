"""Score and classify candidate resumes against a job description."""

from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarities(cleaned_jd: str, cleaned_resumes: List[str]) -> List[float]:
    """Compute cosine similarity (0.0 to 1.0) between JD and each cleaned resume."""
    if not cleaned_resumes:
        return []

    non_empty_indices = [i for i, txt in enumerate(cleaned_resumes) if txt.strip()]
    scores = [0.0] * len(cleaned_resumes)

    if not cleaned_jd.strip() or not non_empty_indices:
        return scores

    corpus = [cleaned_jd] + [cleaned_resumes[i] for i in non_empty_indices]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(jd_vec, resume_vecs).flatten()

    for idx_pos, orig_idx in enumerate(non_empty_indices):
        scores[orig_idx] = float(np.clip(sims[idx_pos], 0.0, 1.0))

    return scores


def score_candidate(
    tfidf_sim: float,
    matched_must: List[str],
    total_must: int,
    matched_nice: List[str],
    total_nice: int,
    years_exp: float,
    min_exp_years: float,
    shortlist_threshold: float = 65.0,
    consider_threshold: float = 45.0,
) -> Dict[str, object]:
    """Calculate a 0-100 score and determine shortlist/reject status."""
    must_ratio = (len(matched_must) / total_must) if total_must > 0 else 1.0
    nice_ratio = (len(matched_nice) / total_nice) if total_nice > 0 else 0.5

    if min_exp_years > 0:
        exp_ratio = min(years_exp / min_exp_years, 1.0)
    else:
        exp_ratio = 1.0

    # Boost TF-IDF cosine similarity slightly so realistic resume/JD text maps well to 0-100
    scaled_sim = min(tfidf_sim * 2.5, 1.0)

    # Weighted score out of 100:
    # 50% must-have skills + 25% TF-IDF similarity + 15% nice-to-have skills + 10% experience
    raw_score = (
        50.0 * must_ratio
        + 25.0 * scaled_sim
        + 15.0 * nice_ratio
        + 10.0 * exp_ratio
    )

    # Penalty if any must-have skills are missing
    missing_count = max(0, total_must - len(matched_must))
    penalty = 6.0 * missing_count
    final_score = round(float(np.clip(raw_score - penalty, 0.0, 100.0)), 2)

    has_must_have_gap = len(missing_count > 0) if isinstance(missing_count, bool) else (missing_count > 0)

    if final_score >= shortlist_threshold and not has_must_have_gap:
        status = "SHORTLIST"
    elif final_score >= consider_threshold:
        status = "CONSIDER"
    else:
        status = "REJECT"

    return {
        "match_score": final_score,
        "tfidf_similarity": round(tfidf_sim, 4),
        "must_have_coverage": f"{len(matched_must)}/{total_must}",
        "has_must_have_gap": has_must_have_gap,
        "status": status,
    }