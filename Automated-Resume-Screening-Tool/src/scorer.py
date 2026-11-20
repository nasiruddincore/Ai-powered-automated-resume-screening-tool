from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarities(cleaned_jd: str, cleaned_resumes: List[str]) -> List[float]:
    if not cleaned_resumes:
        return []
    scores = [0.0] * len(cleaned_resumes)
    valid_idx = [i for i, t in enumerate(cleaned_resumes) if t.strip()]
    if not cleaned_jd.strip() or not valid_idx:
        return scores
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform([cleaned_jd] + [cleaned_resumes[i] for i in valid_idx])
    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
    for pos, idx in enumerate(valid_idx):
        scores[idx] = float(np.clip(sims[pos], 0.0, 1.0))
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
    must_ratio = (len(matched_must) / total_must) if total_must > 0 else 1.0
    nice_ratio = (len(matched_nice) / total_nice) if total_nice > 0 else 0.5
    exp_ratio = min(years_exp / min_exp_years, 1.0) if min_exp_years > 0 else 1.0
    scaled_sim = min(tfidf_sim * 2.5, 1.0)

    raw = 50.0 * must_ratio + 25.0 * scaled_sim + 15.0 * nice_ratio + 10.0 * exp_ratio
    missing_cnt = max(0, total_must - len(matched_must))
    final_score = round(float(np.clip(raw - 6.0 * missing_cnt, 0.0, 100.0)), 2)

    if final_score >= shortlist_threshold and missing_cnt == 0:
        status = "SHORTLIST"
    elif final_score >= consider_threshold:
        status = "CONSIDER"
    else:
        status = "REJECT"

    return {
        "match_score": final_score,
        "tfidf_similarity": round(tfidf_sim, 4),
        "must_have_coverage": f"{len(matched_must)}/{total_must}",
        "status": status,
    }
