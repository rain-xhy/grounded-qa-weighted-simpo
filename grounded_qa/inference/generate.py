"""Greedy long-document generation with per-task stopping criteria."""
import json
from pathlib import Path

from ..backbone import chat_inputs, load_model
from ..data.claims import structure_audit
from ..data.evidence import prompt_parts
from .boundaries import complete_boundary

DECODING_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "decoding.json"


def predict(args, records):
    import torch
    from transformers import (LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor,
                              RepetitionPenaltyLogitsProcessor, StoppingCriteria, StoppingCriteriaList)
    from longcite_modeling_llama import postprocess
    tok, base, _ = load_model(args)
    cfg = json.loads(DECODING_CONFIG.read_text(encoding="utf-8"))
    stops = [tok.eos_token_id, tok.get_command("<|user|>"), tok.get_command("<|observation|>")]
    for row in records:
        summary = row["dataset"] == "gov_report"
        prompt = row["prompt"]
        sentences, numbered, _ = prompt_parts(prompt)
        # Upstream postprocess reads original sentence IDs, offsets and source text.
        from longcite_modeling_llama import get_prompt
        official, source_sentences, source_numbered = get_prompt(row["context"], row["query"])
        if prompt != official or numbered != source_numbered:
            raise ValueError("Generation input must match the official source/question prompt")
        batch = chat_inputs(tok, prompt + "\n\n" + cfg["summary_instruction" if summary else "qa_instruction"])
        batch = batch.to(base.device)
        n = batch["input_ids"].shape[1]
        if n > args.max_tokens or n + cfg["max_new_tokens"] > base.config.max_position_embeddings:
            raise ValueError("Generation exceeds token budget")
        penalty = RepetitionPenaltyLogitsProcessor(cfg["output_repetition_penalty"])
        no_repeat = NoRepeatNGramLogitsProcessor(cfg["output_no_repeat_ngram_size"])

        class OutputOnly(LogitsProcessor):
            def __call__(self, input_ids, scores):
                suffix = input_ids[:, n:]
                return no_repeat(suffix, penalty(suffix, scores)) if suffix.shape[1] else scores

        class StopAtBoundary(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                suffix = input_ids[0, n:].tolist()
                return torch.tensor([complete_boundary(tok.decode(suffix), len(suffix), summary, cfg)],
                                    device=input_ids.device)

        with torch.inference_mode():
            output = base.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"],
                                   eos_token_id=stops, pad_token_id=tok.pad_token_id,
                                   stopping_criteria=StoppingCriteriaList([StopAtBoundary()]),
                                   logits_processor=LogitsProcessorList([OutputOnly()]))
        tokens = output[0, n:].tolist()
        if tokens and tokens[-1] in stops:
            tokens.pop()
        text = tok.decode(tokens).strip()
        raw = "<" + text if text.startswith("statement>") else text
        parsed = postprocess(raw, source_sentences, numbered)
        yield {**row, "status": "ok", "raw_prediction": raw, "prediction": parsed["answer"],
               "statements": parsed["statements_with_citations"], "structure": structure_audit(raw, sentences)}
