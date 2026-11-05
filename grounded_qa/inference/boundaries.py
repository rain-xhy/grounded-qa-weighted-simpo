def complete_boundary(text,generated,summary,cfg):
    # This is a generation stopping rule, never a rewrite of an existing answer.
    # Require the newly generated suffix itself to end with a complete tag.
    if not text.rstrip().endswith('</statement>'):return False
    count=text.count('</statement>')
    if summary:return count>=cfg['summary_max_statements'] or generated>=cfg['summary_soft_token_limit']
    return count>=cfg['qa_max_statements']
