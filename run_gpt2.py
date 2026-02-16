"""Run GPT-2 inference using our runtime.

Loads the real GPT-2 (124M) weights from HuggingFace, exports the
full model (embeddings + transformer blocks + LM head) through our
pipeline, and generates text via greedy autoregressive decoding.

Everything runs through the compiled C executor — embeddings, attention,
GELU, layernorm, and the final LM head projection.

Dynamic shapes: export once with symbolic seq_len, then rebind for each
new sequence length during generation. This avoids re-exporting (~360ms)
and only re-plans + re-compiles (~7ms) per step. Fold-only ops that
depend on seq_len (like arange for position IDs) are evaluated by a
post-resolution constant folding pass.
"""

import time

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from runtime.exporter import export_pytorch, summary
from runtime.session import Session


def main():
    prompt = "The future of artificial intelligence is"
    max_new_tokens = 30

    # --- Load model and tokenizer ---
    print("Loading GPT-2 (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.config.use_cache = False

    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens: {seq_len}")

    # --- Export with dynamic seq_len ---
    example_ids = torch.tensor([input_ids])

    print(f"\nExporting full GPT-2 with dynamic shapes...")
    t0 = time.perf_counter()
    graph = export_pytorch(
        model, (example_ids,),
        dynamic_dims={'L': [('input_ids', 1)]},
    )
    export_time = time.perf_counter() - t0
    print(f"Export: {export_time:.2f}s")

    # --- Create session (optimize + plan + compile for initial seq_len) ---
    t0 = time.perf_counter()
    session = Session(graph)
    session.create(bindings={'L': seq_len})
    compile_time = time.perf_counter() - t0
    print(f"Optimize + plan + compile: {compile_time:.2f}s")
    print(summary(graph))

    # --- Verify single forward pass against PyTorch ---
    ids_np = np.array([input_ids], dtype=np.int64)
    input_name = graph.inputs[0]
    feed = {input_name: ids_np}

    t0 = time.perf_counter()
    result = session.run(feed)
    runtime_ms = (time.perf_counter() - t0) * 1000
    logits = result[graph.outputs[0]]  # [1, seq_len, vocab_size]

    print(f"\nInference: {runtime_ms:.1f}ms")

    with torch.no_grad():
        torch_logits = model(example_ids).logits[0, -1].numpy()
    our_logits = logits[0, -1]
    max_diff = float(np.max(np.abs(our_logits - torch_logits)))
    print(f"Max logit diff vs PyTorch: {max_diff:.6f}")
    assert np.allclose(our_logits, torch_logits, atol=1e-2), "Logits diverged!"
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

        # Rebind to new sequence length (resolve + plan + compile)
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

    # --- Verify against PyTorch greedy decoding (with timing) ---
    print("Verifying against PyTorch greedy decoding...")
    pt_generated = list(input_ids)
    pt_start = time.perf_counter()
    for _ in range(max_new_tokens):
        ids_t = torch.tensor([pt_generated])
        with torch.no_grad():
            pt_logits = model(ids_t).logits
        pt_generated.append(int(pt_logits[0, -1].argmax()))
    pt_elapsed = time.perf_counter() - pt_start
    pt_tok_per_sec = max_new_tokens / pt_elapsed

    if generated == pt_generated:
        print("Generation matches PyTorch exactly!")
    else:
        # Find first divergence
        for i, (a, b) in enumerate(zip(generated, pt_generated)):
            if a != b:
                print(f"Diverged at token {i}: ours={tokenizer.decode([a])!r} vs PyTorch={tokenizer.decode([b])!r}")
                break
        print(f"Ours:    {tokenizer.decode(generated)}")
        print(f"PyTorch: {tokenizer.decode(pt_generated)}")

    print(f"\nPyTorch eager: {max_new_tokens} tokens in {pt_elapsed:.2f}s ({pt_tok_per_sec:.1f} tok/s)")
    print(f"Ours:          {max_new_tokens} tokens in {gen_elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    speedup = pt_elapsed / gen_elapsed
    print(f"Speedup: {speedup:.2f}x")

    # --- Timing summary ---
    tok_per_sec = max_new_tokens / gen_elapsed
    print(f"\nGeneration: {max_new_tokens} tokens in {gen_elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

    avg_rebind = sum(rebind_times) / len(rebind_times)
    avg_infer = sum(infer_times) / len(infer_times)
    print(f"\nPer-step breakdown (avg over {len(rebind_times)} steps):")
    print(f"  Rebind:    {avg_rebind:.1f}ms")
    print(f"  Inference: {avg_infer:.1f}ms")
    print(f"  Total:     {avg_rebind + avg_infer:.1f}ms")
    print(f"\n  Rebind total:  {sum(rebind_times):.0f}ms ({sum(rebind_times)/gen_elapsed/10:.0f}% of generation)")
    print(f"  Infer total:   {sum(infer_times):.0f}ms ({sum(infer_times)/gen_elapsed/10:.0f}% of generation)")

    # Per-step detail
    print(f"\n  {'step':>4} {'seq_len':>7} {'rebind':>8} {'infer':>8} {'total':>8}")
    print(f"  {'-'*39}")
    for i in range(len(rebind_times)):
        total = rebind_times[i] + infer_times[i]
        print(f"  {i:>4} {len(input_ids)+i:>7} {rebind_times[i]:>7.1f}ms {infer_times[i]:>7.1f}ms {total:>7.1f}ms")


if __name__ == "__main__":
    main()
