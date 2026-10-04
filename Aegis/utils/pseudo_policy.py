"""Explicit release provenance: teacher-only data never implies human review."""

HUMAN_REVIEWED = 'human_screen_passed_v1'
TEACHER_ONLY = 'teacher_original_scores_unreviewed_v1'
RELEASED_QUALITY_STATUSES = (HUMAN_REVIEWED, TEACHER_ONLY)
READY_PSEUDO_STATUSES = ('reused_identical_normalized_text', 'generated_strict_json_v1')


def training_release_ready(rows):
    training = [r for r in rows if r['split'] == 'train']
    statuses = {r.get('pseudo_quality_status') for r in training}
    return (bool(training) and len(statuses) == 1 and statuses <= set(RELEASED_QUALITY_STATUSES)
            and all(r['pseudo_status'] in READY_PSEUDO_STATUSES for r in training))
