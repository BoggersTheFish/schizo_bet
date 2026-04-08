#!/usr/bin/env python3
"""
Train TS-scale LM with file-based train/eval, optional BPE, AMP, LR schedule,
checkpoints, and gradient accumulation.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Any, List, Optional, Tuple

Payload = Optional[dict[str, Any]]


def _require_text_file(path: str, label: str) -> None:
    if path and not os.path.isfile(path):
        ap = os.path.abspath(path)
        print(
            f"Error: {label} file not found: {path!r}\n"
            f"  (resolved from cwd {os.getcwd()} -> {ap})\n"
            f"  Create the file, fix the path, or run without --train-file to use the built-in demo.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    print("ts_scale.train: starting", flush=True)
    p = argparse.ArgumentParser(description="Train TS-scale LM (tension-graph backbone)")
    p.add_argument("--steps", type=int, default=80, help="Total optimizer steps to reach")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--stride", type=int, default=0, help="0 = seq_len (non-overlapping windows)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--window", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=0, help="0 = max(50, steps//20)")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--train-file", type=str, default="", help="Training text file (UTF-8)")
    p.add_argument("--eval-file", type=str, default="", help="Held-out eval text (optional)")
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="If no --eval-file, hold out this fraction from the end of train text",
    )
    p.add_argument("--eval-every", type=int, default=0, help="0 = max(1, steps//10)")
    p.add_argument("--max-train-chars", type=int, default=0, help="0 = read full train file")
    p.add_argument("--tokenizer", type=str, choices=("byte", "bpe"), default="byte")
    p.add_argument("--bpe-vocab-size", type=int, default=8000)
    p.add_argument("--text", type=str, default="", help="If no --train-file, use this string")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--amp", action="store_true", help="autocast + GradScaler (CUDA)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ts_run")
    p.add_argument("--ckpt-every", type=int, default=0, help="Periodic checkpoint; 0 = only last/best")
    p.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from checkpoint .pt (pass same --train-file / tokenizer setup)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import math

    import torch
    import torch.optim as optim

    from ts_scale.checkpointing import (
        apply_checkpoint_payload,
        config_from_checkpoint,
        save_checkpoint,
    )
    from ts_scale.config import TSLMConfig
    from ts_scale.eval_metrics import eval_lm_metrics, training_step_log
    from ts_scale.lr_schedule import warmup_cosine_lr_lambda
    from ts_scale.model import TSLanguageModel
    from ts_scale.text_tokenizer import BPETokenizer, ByteTokenizer, read_text_file

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    use_amp = (
        bool(args.amp)
        and not args.no_amp
        and device.type == "cuda"
        and torch.cuda.is_available()
    )
    amp_dtype = torch.bfloat16 if (
        device.type == "cuda" and torch.cuda.is_bf16_supported()
    ) else torch.float16

    pad = " TS tension graph oscillation moe vq schrodinger collapse manifold "
    resume_payload: Payload = None
    start_step = 0

    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = int(resume_payload.get("step", 0))
        if start_step >= args.steps:
            print(
                f"Checkpoint step {start_step} >= --steps {args.steps}; increase --steps.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- load text ---
    if args.train_file:
        _require_text_file(args.train_file, "Training (--train-file)")
        if args.eval_file:
            _require_text_file(args.eval_file, "Eval (--eval-file)")
        cap = args.max_train_chars if args.max_train_chars > 0 else None
        train_full = read_text_file(args.train_file, cap)
        if args.eval_file:
            train_str = train_full
            eval_str = read_text_file(args.eval_file)
        elif args.eval_fraction > 0 and len(train_full) > args.seq_len * 20:
            cut = int(len(train_full) * (1.0 - args.eval_fraction))
            cut = max(cut, args.seq_len * 10)
            train_str = train_full[:cut]
            eval_str = train_full[cut:]
        else:
            train_str = train_full
            eval_str = train_full[-min(len(train_full), args.seq_len * 8) :]
    elif args.text:
        train_str = args.text
        eval_str = args.text[-min(len(args.text), args.seq_len * 8) :]
    elif args.stdin:
        train_str = sys.stdin.read()
        eval_str = train_str[-min(len(train_str), args.seq_len * 8) :]
    else:
        train_str = pad * 8
        eval_str = train_str

    if len(train_str) < args.seq_len + 10:
        train_str = (train_str + pad) * 8
    if len(eval_str) < args.seq_len + 10:
        eval_str = (eval_str + pad) * 4

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer_path: Optional[str] = None
    meta_path = os.path.join(args.checkpoint_dir, "tokenizer_meta.json")

    if resume_payload is not None:
        cfg = config_from_checkpoint(resume_payload)
        tt = resume_payload.get("tokenizer_type", "byte")
        tokenizer_path = resume_payload.get("tokenizer_path")
        if tokenizer_path and not os.path.isfile(tokenizer_path) and args.resume:
            alt = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "tokenizer.json")
            if os.path.isfile(alt):
                tokenizer_path = alt
        if tt == "bpe" and tokenizer_path and os.path.isfile(tokenizer_path):
            tok = BPETokenizer.load(tokenizer_path)
        else:
            tok = ByteTokenizer()
        print(
            f"resume step={start_step} tokenizer={tok.name} vocab={tok.vocab_size}",
            flush=True,
        )
    elif args.tokenizer == "bpe":
        tpath = os.path.join(args.checkpoint_dir, "tokenizer.json")
        tok = BPETokenizer.train_on_text(
            train_str,
            vocab_size=args.bpe_vocab_size,
            save_path=tpath,
        )
        tokenizer_path = tpath
        tok.save_meta(meta_path)
        cfg = TSLMConfig(
            vocab_size=tok.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            window=args.window,
            max_seq_len=args.seq_len + 8,
            vq_num_codes=min(512, max(128, tok.vocab_size // 2)),
        )
    else:
        tok = ByteTokenizer()
        tok.save_meta(meta_path)
        cfg = TSLMConfig(
            vocab_size=tok.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            window=args.window,
            max_seq_len=args.seq_len + 8,
            vq_num_codes=min(512, max(128, tok.vocab_size // 2)),
        )

    stride = args.stride if args.stride > 0 else args.seq_len

    def ids_to_batches(
        ids: List[int],
        seq_len: int,
        stride_: int,
        dev: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        data = torch.tensor(ids, dtype=torch.long)
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        lim = len(data) - seq_len - 1
        i = 0
        while i <= lim:
            chunk = data[i : i + seq_len + 1]
            if chunk.numel() < seq_len + 1:
                break
            inp = chunk[:-1].unsqueeze(0).to(dev)
            lab = chunk[1:].unsqueeze(0).to(dev)
            out.append((inp, lab))
            i += stride_
        return out

    train_ids = tok.encode(train_str)
    eval_ids = tok.encode(eval_str)
    print(
        f"device={device} amp={use_amp} tokenizer={tok.name} vocab={tok.vocab_size} "
        f"train_tok={len(train_ids)} eval_tok={len(eval_ids)}",
        flush=True,
    )

    train_batches = ids_to_batches(train_ids, args.seq_len, stride, device)
    eval_batches = ids_to_batches(eval_ids, args.seq_len, stride, device)
    if not train_batches:
        print("Not enough tokens for one training batch.", file=sys.stderr)
        sys.exit(1)
    if not eval_batches:
        eval_batches = train_batches[:1]

    print(
        f"batches train={len(train_batches)} eval={len(eval_batches)} stride={stride}",
        flush=True,
    )

    model = TSLanguageModel(cfg).to(device)
    opt = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup = (
        args.warmup_steps
        if args.warmup_steps > 0
        else max(50, args.steps // 20)
    )
    lr_fn = warmup_cosine_lr_lambda(
        warmup_steps=warmup,
        total_steps=max(args.steps, warmup + 1),
        min_lr_ratio=args.min_lr_ratio,
    )
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn, last_epoch=-1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume_payload is not None:
        apply_checkpoint_payload(
            resume_payload,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
        )

    eval_every = args.eval_every if args.eval_every > 0 else max(1, args.steps // 10)
    ckpt_every = args.ckpt_every
    best_eval_nll = float("inf")

    def run_eval():
        return eval_lm_metrics(model, eval_batches, device)

    def save_ckpt(name: str, completed_steps: int, extra: Optional[dict] = None) -> None:
        path = os.path.join(args.checkpoint_dir, name)
        save_checkpoint(
            path,
            step=completed_steps,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            cfg=cfg,
            tokenizer_type=tok.name,
            tokenizer_path=tokenizer_path,
            meta_extra=extra,
        )
        print(f"saved {path} step={completed_steps}", flush=True)

    t0 = time.time()
    model.train()
    opt_step = start_step
    data_i = 0
    last_out: Optional[dict[str, Any]] = None

    while opt_step < args.steps:
        accum_loss: Optional[torch.Tensor] = None
        for _ in range(args.grad_accum):
            ids, labels = train_batches[data_i % len(train_batches)]
            data_i += 1
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if use_amp else torch.float32,
                enabled=use_amp,
            ):
                out = model(ids, labels=labels, return_loss_breakdown=True)
                chunk = out["loss"] / float(args.grad_accum)
            last_out = out
            accum_loss = chunk if accum_loss is None else accum_loss + chunk

        assert accum_loss is not None
        if use_amp:
            scaler.scale(accum_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            accum_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        scheduler.step()

        opt_step += 1
        if last_out is not None and (opt_step % 20 == 0 or opt_step == args.steps):
            row = training_step_log(last_out)
            row["lr"] = scheduler.get_last_lr()[0]
            print(opt_step, row, flush=True)

        if opt_step % eval_every == 0 or opt_step == args.steps:
            ev = run_eval()
            sat = "  [PPL display capped; mean_nll_nats > 20]" if ev["ppl_saturated"] else ""
            print(
                f"eval mean_nll_nats={ev['mean_nll_nats']:.4f} "
                f"ppl_capped={ev['ppl_capped']:.4f} @ opt_step {opt_step}{sat}",
                flush=True,
            )
            if math.isfinite(ev["mean_nll_nats"]) and ev["mean_nll_nats"] < best_eval_nll:
                best_eval_nll = ev["mean_nll_nats"]
                save_ckpt(
                    "best.pt",
                    opt_step,
                    {
                        "mean_nll_nats": ev["mean_nll_nats"],
                        "ppl_capped": ev["ppl_capped"],
                    },
                )

        if ckpt_every > 0 and opt_step % ckpt_every == 0:
            save_ckpt(f"step_{opt_step}.pt", opt_step)

    final_ev = run_eval()
    dt = time.time() - t0
    print(
        f"final mean_nll_nats={final_ev['mean_nll_nats']:.4f} "
        f"ppl_capped={final_ev['ppl_capped']:.4f} "
        f"best_mean_nll_nats={best_eval_nll:.4f} time_s={round(dt, 1)}",
        flush=True,
    )
    save_ckpt(
        "last.pt",
        args.steps,
        {
            "final_mean_nll_nats": final_ev["mean_nll_nats"],
            "final_ppl_capped": final_ev["ppl_capped"],
        },
    )


if __name__ == "__main__":
    main()
