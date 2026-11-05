"""Six task metrics from frozen predictions/scores; no judge calls or GPU use.

Citation length counts the parsed source snippets with one shared tokenizer.
It is citation-weighted and is not the mean of per-answer citation lengths.
"""
import math


def latest(rows):
    return {row['id']: row for row in rows}


def average(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return sum(values)/len(values) if values else None


def summarize(predictions, scores, encode, tokenizer_identity):
    predictions, scores = latest(predictions), latest(scores)
    successful = [r for r in predictions.values() if r['status'] == 'ok']
    judged = [scores[r['id']] for r in successful if r['id'] in scores and scores[r['id']].get('semantic_status') == 'ok']
    lengths, empty_spans, uncited_answers = [], 0, 0
    structural = []
    for row in successful:
        structure = row.get('structure', {}).get('structure_valid')
        if structure is None:
            structure = scores.get(row['id'], {}).get('structure_valid')
        structural.append({'structure_valid_rate': structure})
        count = 0
        for claim in row.get('statements', []):
            for citation in claim['citation']:
                text = citation['cite']
                if not isinstance(text, str):
                    raise ValueError('Citation source text is not a string')
                if not text.strip():
                    empty_spans += 1
                    continue
                tokens = encode(text)
                if not tokens:
                    raise ValueError('A nonempty citation produced no tokens')
                lengths.append(len(tokens)); count += 1
        uncited_answers += int(count == 0)
    return {'metrics': {
                'answer_correctness': average(judged, 'answer_correctness'),
                'structure_valid_rate': average(structural, 'structure_valid_rate'),
                'citation_f1': average(judged, 'citation_f1'),
                'citation_recall': average(judged, 'citation_recall'),
                'citation_precision': average(judged, 'citation_precision'),
                'citation_length_tokens_mean': sum(lengths)/len(lengths) if lengths else None},
            'coverage': {'expected_prediction_ids': len(predictions), 'generated': len(successful),
                         'judged': len(judged), 'unjudged_or_errors': len(successful)-len(judged),
                         'semantic_metric_counts': {k: sum(isinstance(r.get(k), (int, float)) for r in judged)
                              for k in ['answer_correctness', 'citation_f1', 'citation_recall', 'citation_precision']},
                         'structure_checked': sum(r['structure_valid_rate'] is not None for r in structural),
                         'citation_spans': len(lengths), 'empty_citation_spans': empty_spans,
                         'generated_answers_without_nonempty_citations': uncited_answers},
            'citation_length_tokenizer': tokenizer_identity,
            'units': 'Quality metrics are 0-1 scores/rates; citation length is tokens per nonempty source span',
            'notes': ['A legal citation ID is not proof of semantic support.',
                      'No citations yields null mean length, never a fabricated zero.',
                      'Semantic coverage and citation-length coverage are reported separately.',
                      'Use the same fixed question IDs and length tokenizer for method comparisons.'],
            'new_paid_api_calls': 0, 'new_gpu_calls': 0}
