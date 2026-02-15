"""Run GPT-2 inference using our runtime.

Loads the real GPT-2 (124M) weights from HuggingFace, exports the
full model (embeddings + transformer blocks + LM head) through our
pipeline, and generates text via greedy autoregressive decoding.

Everything runs through the compiled C executor — embeddings, attention,
GELU, layernorm, and the final LM head projection.

Without KV caching, each generation step re-processes the full sequence.
This means re-exporting the graph for each new sequence length (since
our graph has static shapes). Slow, but demonstrates correctness.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from runtime.exporter import export_pytorch, summary
from runtime.session import Session


class GPT2ForInference(nn.Module):
    """Full GPT-2 for next-token prediction.

    Wraps the HuggingFace model into a clean module that takes
    (input_ids, position_ids) and returns logits. Avoids HuggingFace's
    generate() infrastructure so torch.export gets a clean trace.
    """
    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.wte = model.transformer.wte
        self.wpe = model.transformer.wpe
        self.h = model.transformer.h
        self.ln_f = model.transformer.ln_f
        self.lm_head = model.lm_head

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            hidden = block(hidden)[0]
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


def main():
    prompt = "The future of artificial intelligence is"

    # --- Load model and tokenizer ---
    print("Loading GPT-2 (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()

    model = GPT2ForInference(hf_model)

    input_ids = tokenizer.encode(prompt)
    seq_len = len(input_ids)
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens: {seq_len}")

    # --- Export full model ---
    example_ids = torch.tensor([input_ids])
    example_pos = torch.arange(seq_len).unsqueeze(0)

    print(f"\nExporting full GPT-2...")
    t0 = time.perf_counter()
    graph = export_pytorch(model, (example_ids, example_pos))
    export_time = time.perf_counter() - t0
    print(f"Export: {export_time:.2f}s")

    # --- Create session (optimize + plan + compile) ---
    t0 = time.perf_counter()
    session = Session(graph)
    session.create()
    compile_time = time.perf_counter() - t0
    print(f"Optimize + plan + compile: {compile_time:.2f}s")
    print(summary(graph))

    # --- Single forward pass ---
    ids_np = np.array([input_ids], dtype=np.int64)
    pos_np = np.arange(seq_len, dtype=np.int64)[np.newaxis]
    input_names = graph.inputs

    t0 = time.perf_counter()
    result = session.run({input_names[0]: ids_np, input_names[1]: pos_np})
    runtime_ms = (time.perf_counter() - t0) * 1000
    logits = result[graph.outputs[0]]  # [1, seq_len, vocab_size]

    next_token = int(np.argmax(logits[0, -1]))
    print(f"\nInference: {runtime_ms:.1f}ms")
    print(f"Next token: \"{tokenizer.decode([next_token])}\"")

    # --- Verify against PyTorch ---
    with torch.no_grad():
        torch_logits = hf_model(example_ids).logits[0, -1].numpy()
    our_logits = logits[0, -1]

    max_diff = float(np.max(np.abs(our_logits - torch_logits)))
    print(f"Max logit diff vs PyTorch: {max_diff:.6f}")
    assert np.allclose(our_logits, torch_logits, atol=1e-2), "Logits diverged!"
    print("Correctness verified.")

    # Show what the model would say
    top5 = np.argsort(our_logits)[-5:][::-1]
    print(f"\nTop 5 predictions:")
    for tok in top5:
        print(f"  \"{tokenizer.decode([tok])}\" (logit={our_logits[tok]:.2f})")


if __name__ == "__main__":
    main()
