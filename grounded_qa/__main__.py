"""Run portable training, inference and evaluation stages on user-supplied files."""
import argparse
import importlib
import json

from .io import read_rows, write_rows


def main():
    parser = argparse.ArgumentParser(prog="python -m grounded_qa")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prompt", "trace", "pairs", "sft", "simpo", "intervene", "predict", "judge", "metrics"):
        p = subs.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--out", required=True)
        if name in ("sft", "simpo", "trace", "predict"):
            p.add_argument("--model", required=True, help="Local base model directory")
            p.add_argument("--tokenizer", help="Local tokenizer directory if separate from base")
            p.add_argument("--adapter", help="Local input adapter")
            p.add_argument("--max-tokens", type=int, default=16384)
            p.add_argument("--seed", type=int, default=42)
        if name in ("sft", "simpo"):
            p.add_argument("--lr", type=float, required=True)
            p.add_argument("--epochs", type=int, default=1)
            p.add_argument("--accumulate", type=int, default=4)
        if name == "intervene":
            p.add_argument("--axis", required=True, choices=list("QECB"))
        if name == "metrics":
            p.add_argument("--scores", required=True)
            p.add_argument("--tokenizer", required=True)
        if name == "judge":
            p.add_argument("--scorer-dir", required=True)
            p.add_argument("--judge-model", required=True)
            p.add_argument("--transport", required=True, help="Trusted module:factory returning an object with query")
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.command == "prompt":
        from longcite_modeling_llama import get_prompt
        def attach(row):
            prompt, sentences, _ = get_prompt(row["context"], row["query"])
            return {**row, "prompt": prompt, "sentence_count": len(sentences)}
        write_rows(args.out, map(attach, rows))
    elif args.command == "pairs":
        from .preferences import make_pair
        write_rows(args.out, (pair for row in rows if (pair := make_pair(row)) is not None))
    elif args.command == "intervene":
        for row in rows:
            if row["axis"] not in ("Q", "E", "C", "B") or row["preferred"] == row["weakened"]:
                raise ValueError("Expected a tagged contrast between two different targets")
        write_rows(args.out, ({"id": row["id"], "prompt": row["prompt"],
                               "response": row["weakened" if row["axis"] == args.axis else "preferred"]}
                              for row in rows))
    elif args.command == "metrics":
        from pathlib import Path
        from transformers import AutoTokenizer
        from .metrics import summarize
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
        result = summarize(rows, read_rows(args.scores), lambda t: tok.encode(t, add_special_tokens=False),
                           Path(args.tokenizer).name)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif args.command == "judge":
        from .judge import author_evaluator, semantic_scores
        mod, factory = args.transport.rsplit(":", 1)
        judge = getattr(importlib.import_module(mod), factory)()
        cite = author_evaluator(args.scorer_dir, "auto_scorer.py", judge.query, args.judge_model)
        correct = author_evaluator(args.scorer_dir, "eval_correct.py", judge.query, args.judge_model)
        def scores():
            for row in rows:
                if row["status"] == "ok":
                    yield {"id": row["id"], "semantic_status": "ok",
                           **semantic_scores(row, cite, correct, judge, args.judge_model)}
        write_rows(args.out, scores())
    else:
        from .model import predict, trace, train
        if args.command in ("sft", "simpo"):
            train(args, rows)
        else:
            write_rows(args.out, (trace if args.command == "trace" else predict)(args, rows))


if __name__ == "__main__":
    main()
