import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    index: int
    text: str
    citation: str
    sentence_ids: tuple


PATTERN = re.compile(r'<statement>(.*?)<cite>(.*?)</cite></statement>', re.S)


def parse_claims(raw, sentence_count):
    claims, errors = [], []
    matches = list(PATTERN.finditer(raw))
    if not matches or PATTERN.sub('', raw).strip():
        errors.append('unparsed_answer_text_or_missing_statement_tags')
    for index, match in enumerate(matches):
        text, cite = match.group(1).strip(), match.group(2).strip()
        if not text:
            errors.append(f'empty_statement:{index}')
        ranges = re.findall(r'\[(\d+)-(\d+)\]', cite)
        if re.sub(r'\[\d+-\d+\]', '', cite).strip():
            errors.append(f'malformed_citation:{index}')
        ids = set()
        for start, end in ranges:
            start, end = int(start), int(end)
            if not 0 <= start <= end < sentence_count:
                errors.append(f'citation_out_of_range:{index}')
            else:
                ids.update(range(start, end+1))
        claims.append(Claim(index, text, cite, tuple(sorted(ids))))
    return claims, errors


def structure_audit(raw, sentences):
    """Validate raw citations BEFORE upstream parsing (which clips out-of-range IDs)."""
    total, valid, malformed = 0, 0, 0
    for block in re.findall(r'<cite>(.*?)</cite>', raw, flags=re.S):
        ranges = re.findall(r'\[(\d+)-(\d+)\]', block)
        residual = re.sub(r'\[\d+-\d+\]', '', block).strip()
        malformed += int(bool(residual))
        for start, end in ranges:
            total += 1
            valid += int(0 <= int(start) <= int(end) < len(sentences))
    tags_balanced = (raw.count('<statement>') == raw.count('</statement>') and
                     raw.count('<cite>') == raw.count('</cite>'))
    return {'citation_ranges': total, 'valid_ranges': valid,
            'invalid_ranges': total - valid, 'malformed_citation_blocks': malformed,
            'tags_balanced': tags_balanced,
            'has_statement': '<statement>' in raw,
            'structure_valid': tags_balanced and '<statement>' in raw and
                               total == valid and malformed == 0}


def plain_answer(text):
    text = re.sub(r'<cite>.*?</cite>', '', text, flags=re.S)
    return text.replace('<statement>', '').replace('</statement>', '').strip()
