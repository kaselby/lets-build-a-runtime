"""Run Qwen3 inference using our runtime.

Loads real Qwen3-4B weights from HuggingFace, exports the full model
(embeddings + transformer blocks + LM head) through our pipeline, and
generates text via greedy autoregressive decoding.

Everything runs through the compiled C executor — embeddings, RoPE,
GQA attention, RMSNorm, SiLU gated MLP, and the final LM head.

Dynamic shapes: export once with symbolic seq_len, then rebind for each
new sequence length during generation.

Usage:
    python run_qwen3.py [--model Qwen/Qwen3-0.6B] [--prompt "..."] [--tokens 30]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.exporter import export_pytorch, summary
from runtime.session import Session


class CausalLMWrapper(nn.Module):
    """Wrapper that returns logits tensor directly (no CausalLMOutput)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids, use_cache=False).logits


def main():
    parser = argparse.ArgumentParser(description="Run Qwen3 inference")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-4B)")
    parser.add_argument("--prompt", type=str,
                        default="The future of artificial intelligence is",
                        help="Prompt to complete")
    parser.add_argument("--tokens", type=int, default=30,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    prompt = args.prompt
    max_new_tokens = args.tokens

    # --- Load model and tokenizer ---
    print(f"Loading {args.model}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False
    load_time = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.parameters())
    weight_gb = n_params * 4 / 1e9
    print(f"  {n_params/1e9:.1f}B parameters ({weight_gb:.1f} GB fp32)")
    print(f"  Loaded in {load_time:.1f}s")

    wrapper = CausalLMWrapper(model)
    wrapper.eval()

    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)
    print(f"\nPrompt: \"{prompt}\"")
    print(f"Tokens: {seq_len}")

    # --- Export with dynamic seq_len ---
    example_ids = torch.tensor([input_ids])

    print(f"\nExporting with dynamic shapes...")
    t0 = time.perf_counter()
    graph = export_pytorch(
        wrapper, (example_ids,),
        dynamic_dims={'L': [('input_ids', 1)]},
    )
    export_time = time.perf_counter() - t0
    print(f"Export: {export_time:.1f}s")

    # --- Create session ---
    t0 = time.perf_counter()
    session = Session(graph)
    session.create(bindings={'L': seq_len})
    compile_time = time.perf_counter() - t0
    print(f"Optimize + plan + compile: {compile_time:.1f}s")
    print(summary(graph))

    # --- Verify single forward pass against PyTorch ---
    ids_np = np.array([input_ids], dtype=np.int64)
    input_name = graph.inputs[0]
    feed = {input_name: ids_np}

    t0 = time.perf_counter()
    result = session.run(feed)
    runtime_ms = (time.perf_counter() - t0) * 1000
    logits = result[graph.outputs[0]]

    print(f"\nInference: {runtime_ms:.1f}ms")

    with torch.no_grad():
        torch_logits = wrapper(example_ids)[0, -1].numpy()
    our_logits = logits[0, -1]
    max_diff = float(np.max(np.abs(our_logits - torch_logits)))
    print(f"Max logit diff vs PyTorch: {max_diff:.6f}")
    assert np.allclose(our_logits, torch_logits, atol=1e-2), \
        f"Logits diverged! Max diff: {max_diff}"
    print("Correctness verified.\n")

    # --- Autoregressive generation ---
    print(f"Generating {max_new_tokens} tokens...")
    print(f"  {prompt}", end="", flush=True)

    generated = list(input_ids)
    rebind_times = []
    infer_times = []

    gen_start = time.perf_counter()
    for step in range(max_new_tokens):
        cur_len = len(generated)

        # Rebind to new sequence length
        t0 = time.perf_counter()
        session.rebind({'L': cur_len})
        rebind_ms = (time.perf_counter() - t0) * 1000
        rebind_times.append(rebind_ms)

        # Run inference
        ids_np = np.array([generated], dtype=np.int64)
        feed = {input_name: ids_np}
        t0 = time.perf_counter()
        result = session.run(feed)
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)
        logits = result[graph.outputs[0]]

        # Greedy decode
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)

    gen_elapsed = time.perf_counter() - gen_start
    tok_per_sec = max_new_tokens / gen_elapsed
    print("\n")

    # --- Verify against PyTorch greedy decoding ---
    print("Verifying against PyTorch greedy decoding...")
    pt_generated = list(input_ids)
    pt_start = time.perf_counter()
    for _ in range(max_new_tokens):
        ids_t = torch.tensor([pt_generated])
        with torch.no_grad():
            pt_logits = wrapper(ids_t)
        pt_generated.append(int(pt_logits[0, -1].argmax()))
    pt_elapsed = time.perf_counter() - pt_start
    pt_tok_per_sec = max_new_tokens / pt_elapsed

    if generated == pt_generated:
        print("Generation matches PyTorch exactly!")
    else:
        for i, (a, b) in enumerate(zip(generated, pt_generated)):
            if a != b:
                print(f"Diverged at token {i}: ours={tokenizer.decode([a])!r} "
                      f"vs PyTorch={tokenizer.decode([b])!r}")
                break
        print(f"Ours:    {tokenizer.decode(generated)}")
        print(f"PyTorch: {tokenizer.decode(pt_generated)}")

    print(f"\nPyTorch eager: {max_new_tokens} tokens in {pt_elapsed:.2f}s "
          f"({pt_tok_per_sec:.1f} tok/s)")
    print(f"Ours:          {max_new_tokens} tokens in {gen_elapsed:.2f}s "
          f"({tok_per_sec:.1f} tok/s)")
    speedup = pt_elapsed / gen_elapsed
    print(f"Speedup: {speedup:.2f}x")

    # --- Timing summary ---
    avg_rebind = sum(rebind_times) / len(rebind_times)
    avg_infer = sum(infer_times) / len(infer_times)
    print(f"\nPer-step breakdown (avg over {len(rebind_times)} steps):")
    print(f"  Rebind:    {avg_rebind:.1f}ms")
    print(f"  Inference: {avg_infer:.1f}ms")
    print(f"  Total:     {avg_rebind + avg_infer:.1f}ms")
    print(f"\n  Rebind total:  {sum(rebind_times):.0f}ms "
          f"({sum(rebind_times)/gen_elapsed/10:.0f}% of generation)")
    print(f"  Infer total:   {sum(infer_times):.0f}ms "
          f"({sum(infer_times)/gen_elapsed/10:.0f}% of generation)")

    print(f"\n  {'step':>4} {'seq_len':>7} {'rebind':>8} {'infer':>8} {'total':>8}")
    print(f"  {'-'*39}")
    for i in range(len(rebind_times)):
        total = rebind_times[i] + infer_times[i]
        print(f"  {i:>4} {len(input_ids)+i:>7} "
              f"{rebind_times[i]:>7.1f}ms {infer_times[i]:>7.1f}ms {total:>7.1f}ms")


if __name__ == "__main__":
    main()
