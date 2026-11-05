"""Train SFT and SimPO adapters with window-normalized gradients."""
import math
import random
from pathlib import Path

from ..backbone import encoded_loss, load_model
from .weighted_simpo import backward_weighted_pair, validate_weights


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
