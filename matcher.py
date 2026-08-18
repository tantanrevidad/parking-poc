"""
matcher.py
----------
Confidence-weighted plate-to-ticket matching.

Given a (possibly noisy) OCR read of a plate and a small candidate pool of
"currently checked-in, payment-settled" tickets, score each candidate by an
edit-distance where substitution cost is reduced for low-confidence
characters and for commonly-confused character pairs (0/O, 1/I, 8/B, 5/S,
2/Z). A match is only accepted if the top score clears an absolute
threshold AND has a clear margin over the runner-up — otherwise the slot is
left "unresolved" rather than guessing.

This is real matching logic; only the input data (OCR reads, ticket pool)
is synthetic.
"""

import ast

CONFUSABLE_PAIRS = {
    frozenset({"0", "O"}),
    frozenset({"1", "I"}),
    frozenset({"8", "B"}),
    frozenset({"5", "S"}),
    frozenset({"2", "Z"}),
}

ACCEPT_THRESHOLD = 0.80   # minimum normalized score to accept a match
MIN_MARGIN = 0.08         # required gap between best and second-best score


def _is_confusable(a, b):
    return frozenset({a, b}) in CONFUSABLE_PAIRS


def weighted_similarity(read_text, char_confidences, candidate_plate):
    """
    Returns a similarity score in [0,1] between an OCR read (with
    per-character confidences) and a candidate plate string.

    Same-length comparison (plates are fixed-format in this POC); a
    real system would also handle length mismatches via edit distance.
    """
    if len(read_text) != len(candidate_plate):
        # Heavily penalize length mismatch, but don't hard-fail — a real
        # system might still see this if a character was dropped/inserted.
        return 0.0

    total_cost = 0.0
    max_cost = len(read_text)

    for i, (r_ch, c_ch) in enumerate(zip(read_text, candidate_plate)):
        conf = char_confidences[i] if i < len(char_confidences) else 0.5
        if r_ch == c_ch:
            cost = 0.0
        elif _is_confusable(r_ch, c_ch):
            # confusable mismatch: cheap, and cheaper still if the OCR was
            # already unsure about this character
            cost = 0.25 * (1 - conf * 0.5)
        else:
            # genuine mismatch: cost scales with how CONFIDENT the OCR was
            # (a confident wrong read is more damning than an unsure one)
            cost = 0.5 + 0.5 * conf
        total_cost += cost

    similarity = 1.0 - (total_cost / max_cost)
    return max(0.0, similarity)


def match_plate(read_text, char_confidences, candidate_pool):
    """
    candidate_pool: list of dicts with at least {"ticket_id", "plate"}.
    Returns a dict describing the outcome:
        {
          "resolved": bool,
          "matched_ticket_id": str | None,
          "ranked_candidates": [ (ticket_id, plate, score), ... ]  # sorted desc
        }
    """
    if isinstance(char_confidences, str):
        char_confidences = ast.literal_eval(char_confidences)

    scored = []
    for cand in candidate_pool:
        score = weighted_similarity(read_text, char_confidences, cand["plate"])
        scored.append((cand["ticket_id"], cand["plate"], round(score, 4)))

    scored.sort(key=lambda x: x[2], reverse=True)

    if not scored:
        return {"resolved": False, "matched_ticket_id": None, "ranked_candidates": []}

    best = scored[0]
    second = scored[1][2] if len(scored) > 1 else 0.0
    margin = best[2] - second

    resolved = best[2] >= ACCEPT_THRESHOLD and margin >= MIN_MARGIN

    return {
        "resolved": resolved,
        "matched_ticket_id": best[0] if resolved else None,
        "ranked_candidates": scored,
    }