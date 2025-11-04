# benchmark_vllm_n_offline.py
import argparse, time, statistics, concurrent.futures as cf
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

def count_output_tokens_from_outputs(outputs) -> int:
    # vLLM returns List[RequestOutput], each with .outputs (list of SequenceOutput)
    # Count only generated tokens (excludes prompt tokens).
    total = 0
    for req in outputs:
        for seq in req.outputs:
            total += len(seq.token_ids)
    return total
    
def bench_batched(
    llm: LLM,
    prompts: List[str],
    n: int,
    max_tokens: int,
    temperature: float
) -> Dict[str, Any]:
    # Single call with many prompts → captures true batch wall-time.
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )
    t1 = time.perf_counter()
    wall = t1 - t0
    out_tokens = count_output_tokens_from_outputs(outputs)
    num_reqs = len(prompts)
    # In a single batch call we don’t have per-request latencies; report total wall and simple per-req proxy.
    per_req_proxy = wall / num_reqs if num_reqs else 0.0
    return {
        "mode": "batched",
        "n": n,
        "num_reqs": num_reqs,
        "batch_latency_s": wall,
        "per_req_latency_proxy_s": per_req_proxy,
        "throughput_req_per_s": (num_reqs / wall) if wall > 0 else 0.0,
        "throughput_tok_per_s": (out_tokens / wall) if wall > 0 else 0.0,
        "total_output_tokens": out_tokens,
    }
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    ap.add_argument("--prompts_file", required=False, help="Text file with one prompt per line.")
    ap.add_argument("--num_prompts", type=int, default=32, help="If no file, synthesize this many mathy prompts.")
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    # Build prompts
    if args.prompts_file:
        import pandas as pd
        import re
        dataset = pd.read_parquet(args.prompts_file)

        prompts = []
        print(f"Dataset len {len(dataset)}")
        for i in range(len(dataset)):
            if i == 0:
                print("<｜User｜>"+dataset.iloc[i]['prompt'][0]['content']+"<｜Assistant｜><think>")
            prompts.append("<｜User｜>"+dataset.iloc[i]['prompt'][0]['content']+"<｜Assistant｜><think>")



    # Initialize the engine once (vLLM will pin the model in GPU memory)
    llm = LLM(model=args.model)

    print(f"# Model: {args.model}")
    print(f"# Prompts: {len(prompts)} | Concurrency: {args.concurrency} | max_tokens: {args.max_tokens} | temperature: {args.temperature}")

    metrics = bench_batched(
        llm=llm,
        prompts=prompts,
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print("\n=== RESULTS ===")
    # Pretty print, keeping ints as ints
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
