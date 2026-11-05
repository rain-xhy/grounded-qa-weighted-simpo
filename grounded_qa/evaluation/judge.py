import ast
import copy
import hashlib
import json
import os
import random
import re
from pathlib import Path

import numpy as np
from ..data.claims import plain_answer


def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def author_evaluator(directory, filename, query, model):
    """Use author functions/prompts; replace only network transport and model name."""
    path = Path(directory) / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            if all(t.id not in {'pred_paths'} for t in node.targets):
                nodes.append(node)
    namespace = {'query_llm': query, 'np': np, 're': re, 'random': random,
                 'json': json, 'os': os, 'traceback': __import__('traceback')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['GPT_MODEL'] = model
    return namespace


def semantic_scores(record, cite_module, correct_module, judge, judge_model):
    result = {}
    # Keep upstream clipping/three-spans behaviour for comparability; raw failures
    # are reported separately and never erased by these semantic metrics.
    scored = cite_module['get_citation_score'](copy.deepcopy(record), max_statement_num=40)
    result.update({k: float(scored[k]) for k in ['citation_recall', 'citation_precision', 'citation_f1']})
    result['citation_statements'] = scored['statements']
    prediction = plain_answer(record['prediction'])
    references = record.get('answer') or []
    expected = record.get('expected_decision')
    if expected == 'insufficient':
        references = ['The document does not provide sufficient information to answer the question.']
    elif expected == 'conflict':
        references = ['Explain that the sources conflict, report the incompatible positions, and do not '
                      'arbitrarily choose one. Positions: ' + '; '.join(references)]
    metric = correct_module['dataset2metric'].get(record['dataset'], correct_module['gpt_score_qa'])
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
    if not references:
        result['answer_correctness'] = None
    else:
        random.seed(int(fingerprint(record['id'])[:8], 16))
        values = []
        for reference in references:
            before = len(usage['responses'])
            value = metric(prediction, reference, query=record['query'],
                           few_shot_scores=record.get('few_shot_scores') or [], gpt_usage=usage)
            if len(usage['responses']) == before:
                raise RuntimeError('Author judge parser failed; suppress its fallback score of 0.5')
            values.append(float(value))
        result['answer_correctness'] = max(values)
    if expected is not None:
        prompt = ('Classify the assistant response to this question. Return exactly [[answer]], '
                  '[[insufficient]], or [[conflict]]. answer means it supplies a determinate answer; '
                  'insufficient means it says the requested answer cannot be established from the document; '
                  'conflict means it reports incompatible source answers without arbitrarily resolving them. '
                  'Ignore incidental disclaimers when a determinate answer is still asserted.\nQuestion: ' +
                  record['query'] + '\nResponse: ' + prediction)
        raw = judge.query([{'role': 'user', 'content': prompt}], judge_model, max_new_tokens=30)
        match = re.search(r'\[\[(answer|insufficient|conflict)\]\]', raw)
        if not match:
            raise RuntimeError('Decision judge format failure')
        result['decision'] = match.group(1)
        result['decision_correct'] = float(result['decision'] == expected)
    return result
