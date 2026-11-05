"""Measure the effect of keeping and removing cited source sentences."""
from ..backbone import encoded_loss, load_model
from ..data.claims import PATTERN, parse_claims
from ..data.evidence import evidence_prompt, prompt_parts


def trace(args, records):
    import torch
    tok, base, _ = load_model(args)
    if args.adapter:
        raise ValueError("Contribution uses the frozen base model without an adapter")
    with torch.inference_mode():
        for group in records:
            prompt = group["prompt"]
            sentences, _, _ = prompt_parts(prompt)
            if len(sentences) != group["sentence_count"]:
                raise ValueError("Prompt and source sentence count differ")
            for candidate in group["candidates"]:
                raw = candidate["response"]
                claims, errors = parse_claims(raw, len(sentences))
                if errors:
                    raise ValueError(str(errors))
                traces = []
                for claim, match in zip(claims, PATTERN.finditer(raw)):
                    ids = set(claim.sentence_ids)
                    conditions = {"full": prompt, "keep": evidence_prompt(prompt, sentences, ids, True),
                                  "drop": evidence_prompt(prompt, sentences, ids, False)}
                    traces.append({name: -float(encoded_loss(tok, base, ctx, claim.text, args.max_tokens,
                                                             history=raw[:match.start()], add_eos=False))
                                   for name, ctx in conditions.items()})
                candidate["traces"] = traces
            yield group
