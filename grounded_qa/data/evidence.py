import re


def prompt_parts(prompt):
    _,rest=prompt.split('[Document Start]\n',1);numbered,after=rest.rsplit('\n[Document End]',1)
    markers=list(re.finditer(r'<C(\d+)>',numbered))
    if not markers or [int(m.group(1)) for m in markers]!=list(range(len(markers))):raise ValueError('Nonconsecutive actual source IDs')
    sentences=[{'content':numbered[m.end():(markers[i+1].start() if i+1<len(markers) else len(numbered))]} for i,m in enumerate(markers)]
    return sentences,numbered,after.strip()


def evidence_prompt(prompt, sentences, ids, keep):
    # Preserve original IDs and all retained sentence content. Question is unchanged.
    before, remainder = prompt.split('[Document Start]\n', 1)
    _, after = remainder.rsplit('\n[Document End]', 1)
    context = ''.join(f"<C{i}>" + sentence['content'] for i, sentence in enumerate(sentences)
                      if (i in ids) == keep)
    return before + '[Document Start]\n' + context + '\n[Document End]' + after
