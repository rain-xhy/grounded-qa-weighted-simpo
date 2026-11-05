"""Single-GPU QLoRA, trace scoring, and citation-bounded greedy decoding."""
import json
import math
import random
from pathlib import Path

from .claims import PATTERN, parse_claims, structure_audit
from .decoding import complete_boundary
from .evidence import evidence_prompt, prompt_parts
from .gap_weighted_simpo import backward_weighted_pair, validate_weights
from .response import chat_inputs, memory_adapt, target_token_loss


def load_model(args, training=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from longcite_modeling_llama import LlamaForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("The QLoRA runner requires a CUDA GPU")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, local_files_only=True,
                                              trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    base = LlamaForCausalLM.from_pretrained(args.model, local_files_only=True, device_map={"": 0},
                                           torch_dtype=torch.float16, attn_implementation="sdpa",
                                           quantization_config=quant)
    memory_adapt(base, 512)
    if training:
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.adapter:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=training)
    elif training:
        cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=.05,
                         target_modules=["q_proj", "v_proj"], bias="none")
        model = get_peft_model(base, cfg)
    else:
        model = base
    model.train(training)
    return tokenizer, base, model


def encoded_loss(tokenizer, base, prompt, response, limit, history="", add_eos=True):
    import torch
    prefix = chat_inputs(tokenizer, prompt)["input_ids"][0].tolist()
    prefix += tokenizer.encode(history, add_special_tokens=False)
    target = tokenizer.encode(response, add_special_tokens=False)
    if not target:
        raise ValueError("Empty response")
    if add_eos:
        target.append(tokenizer.eos_token_id)
    if len(prefix) + len(target) > limit:
        raise ValueError("Input and target exceed the token budget")
    ids = torch.tensor([prefix + target], device=base.device)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = base.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            return_dict=True).last_hidden_state[:, len(prefix)-1:-1]
        return target_token_loss(base.lm_head._case5_full_forward, hidden, ids[:, len(prefix):])


def train(args, records):
    import torch
    if not records or args.epochs < 1 or args.accumulate < 1 or args.lr <= 0:
        raise ValueError("Training needs examples and positive training parameters")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Choose a new adapter directory: {out}")
    pairwise = args.command == "simpo"
    if pairwise and not args.adapter:
        raise ValueError("Pass the selected SFT adapter with --adapter")
    if pairwise:
        for row in records:
            gap = row["reward_gap"]
            if not .05 <= gap <= 1 or not math.isclose(row["sample_weight"], 1 + .3 * gap):
                raise ValueError("Invalid preference gap or sample weight")
    tok, base, model = load_model(args, training=True)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
    rng = random.Random(args.seed)
    for _ in range(args.epochs):
        indices = list(range(len(records)))
        rng.shuffle(indices)
        for start in range(0, len(indices), args.accumulate):
            window = [records[i] for i in indices[start:start + args.accumulate]]
            optimizer.zero_grad(set_to_none=True)
            denom = validate_weights([r["sample_weight"] for r in window]) if pairwise else len(window)
            for row in window:
                def loss(response):
                    return encoded_loss(tok, base, row["prompt"], response, args.max_tokens)
                if pairwise:
                    backward_weighted_pair(lambda chosen: loss(row["chosen" if chosen else "rejected"]),
                                           row["sample_weight"], denom, beta=2., gamma_beta_ratio=.1, scaler=scaler)
                else:
                    scaler.scale(loss(row["response"]) / denom).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
    model.save_pretrained(out, safe_serialization=True)


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


def predict(args, records):
    import torch
    from transformers import (LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor,
                              RepetitionPenaltyLogitsProcessor, StoppingCriteria, StoppingCriteriaList)
    from longcite_modeling_llama import postprocess
    tok, base, _ = load_model(args)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "decoding.json").read_text(encoding="utf-8"))
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
