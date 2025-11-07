from __future__ import annotations
import json, argparse, sys, time, csv
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator, Any

from model_clients import call_model, get_embedding
from evaluators import (exact_match, bleu_score as bleu_score_fn, text_embedding_similarity, llm_judge_score)


def parse_args(argv: list[str] | None= None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reading and showing prompts from prompts.jsonl."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to jsonl file with prompt field per line"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model to use for generating the response",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "outputs.jsonl",
        help="Path to write JSONL output results",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Define the max number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Define temperature for level of creativity"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Define top-p for nucleus sampling"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Define repetition penalty (prevent token repetition for diversity) - ONLY WORKS FOR LEGACY APIs"
    )

    #client settings
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds"
        )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for failed requests"
    )

    #flags for evaluation / evaluation settings
    parser.add_argument(
        "--eval",
        action="store_true",
        help="enables eval metrics for lines that have a reference field"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="em,bleu,embed,judge",
        help="Desired metrics to calculate: em, bleu, embed, judge"
    )
    parser.add_argument(
        "--reference-key",
        type=str,
        default="reference",
        help="Key to use for reference text in the input JSONL"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for similarity metric"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model to use for LLM-as-a-judge metric evaluation"
    )
    parser.add_argument(
        "--judge-raw",
        action="store_true",
        help="Print the LLM judge's raw explanation of the score"
    )

    #additional control
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No writing to output files, just print to console"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after a certain number of samples (useful for testing)"
    )
    return parser.parse_args(argv)

def read_jsonl(path: Path) -> Iterator[tuple[int, Any | None]]: # int, Any | None, because of yield lineno, obj
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s=line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s) # parsing json data into an object python can read/understand
            except json.JSONDecodeError as e:
                print(f"Warning: {path}:{lineno} error decoding JSON: {e}. Skipping line {lineno}.", file=sys.stderr)
                yield lineno, None
                continue
            yield lineno, obj

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = args.model
    if not model:
        model = "meta-llama/Meta-Llama-3-8B-Instruct"
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"Error: input path is not a file: {input_path}", file=sys.stderr)
        return 2
    #if not args.dry_run:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    printed = 0
    skipped = 0
    success = 0
    failed = 0

    selected_metrics = set()
    if args.eval:
        for m in (args.metrics or "").split(","):
            m = m.strip().lower()
            if m:
                selected_metrics.add(m)
        allowed = {"em", "bleu", "embed", "judge"}
        unknown = selected_metrics - allowed
        if unknown:
            print(f"Warning: unknown metrics {unknown}, using default metrics {allowed}", file=sys.stderr)
            selected_metrics = allowed

    if args.eval and {"embed", "judge"} & selected_metrics:
        try:
            with input_path.open("r", encoding="utf-8") as f:                        
                total_lines = sum(1 for _ in f)
            if total_lines > 200:
                print(f"Warning: {total_lines} lines detected, embed/judge metrics add cost. Consider removing costly metrics or limit the number of samples", file=sys.stderr)
        except Exception:
            pass
    
    used_temperature = 0.0 if args.eval else float(args.temperature)
    used_top_p = 1.0 if args.eval else float(args.top_p)

    @lru_cache(maxsize=2048)
    def _cached_embed(text: str) -> tuple[float, ...]:
        vec = get_embedding(
            text=text,
            model=args.embed_model,
            timeout=float(args.timeout)
        )
        return tuple(float(x) for x in vec)
    
    def embedder(text: str) -> list[float]:
        return list(_cached_embed(text))
    
    judge_model = args.judge_model or model

    def judge_fn(prompt_text: str) -> str:
        return call_model(
            prompt=prompt_text,
            model=judge_model,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=float(args.repetition_penalty),
            timeout=float(args.timeout),
            retries=int(args.retries)
        )
    
    mode = "a" if args.append else "w"
    out_f = None
    try:
        if not args.dry_run:                                                                              
              output_path.parent.mkdir(parents=True, exist_ok=True)                                         
              mode = "a" if args.append else "w"                                                            
              out_f = output_path.open(mode, encoding="utf-8")

        eval_summary: list[tuple[int, float, dict[str, Any]]] = []
        judge_explanations: list[tuple[int, str]] = []
        summary_rows: list[dict[str, Any]] = []
        for lineno, obj in read_jsonl(input_path):
            if obj is None:
                skipped += 1
                continue
            if not isinstance(obj, dict) or "prompt" not in obj:
                print(f"Warning: {input_path}:{lineno} is missing 'prompt' key, skipping line {lineno}", file=sys.stderr)
                skipped += 1
                continue
            
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"Warning: {input_path}:{lineno} prompt is empty or is not a string, skipping the line {lineno}", file=sys.stderr)
                skipped += 1
                continue

            reference = obj.get(args.reference_key)

            printed += 1
            print(f"\nPrompt {printed} --> " + "'" + prompt + "'")

            if args.dry_run:
                output_text = None
                error_text = None
                elapsed_ms = 0.0
            else:
                started = time.perf_counter()
                output_text: str | None = None
                error_text: str | None = None
                try:
                    output_text = call_model(
                        prompt=prompt,
                        model=model,
                        max_tokens=int(args.max_tokens),
                        temperature=used_temperature,
                        top_p=used_top_p,
                        repetition_penalty=float(args.repetition_penalty),
                        timeout=float(args.timeout),
                        retries=int(args.retries)
                    )
                    success += 1
                    print(f"Response {printed}: {output_text}")
                except Exception as e:
                    error_text = str(e)
                    failed += 1
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(f"Elapsed time: {elapsed_ms:.3f} ms")
            if args.dry_run:                                                                                    
                mode_str = "append" if args.append else "overwrite"                                             
                print(f"[dry-run] would call model={model} "                                     
                    f"max_tokens={args.max_tokens} temp={used_temperature} top_p={used_top_p}; "              
                    f"write -> {output_path} ({mode_str})")
            if output_text is not None and output_text.strip().lower() == "null":                     
                output_text = None 
            metrics: dict[str, Any] = {}
            if (args.eval and reference is not None and isinstance(reference, str) and output_text):
                try:
                    if "em" in selected_metrics:
                        metrics["exact_match"] = exact_match(output_text, reference)
                    if "bleu" in selected_metrics:
                        metrics["bleu"] = bleu_score_fn(output_text, reference, max_n=4)
                    if "embed" in selected_metrics:
                        metrics["embedding_cosine"] = text_embedding_similarity(output_text, reference, embedder)
                    if "judge" in selected_metrics:
                        judge = llm_judge_score(output_text, reference, judge_fn)
                        metrics["judge_score"] = judge.get("score")
                        metrics["judge_raw"] = judge.get("raw")
                except Exception as me:
                    metrics["error"] = f"metric_error: {me}"

            record = {
                "lineno": lineno,
                "prompt": prompt,
                "model": model,
                "output": output_text,
                "error": error_text,
                "latency_ms": round(elapsed_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if reference is not None:
                record[args.reference_key] = reference
            if metrics:                                                                                         
                eval_summary.append((lineno, elapsed_ms, dict(metrics)))
                if args.judge_raw and metrics.get("judge_raw"):
                    judge_explanations.append((lineno, metrics["judge_raw"]))                                              
                record["metrics"] = metrics

            if not args.dry_run and out_f is not None:                                                        
              out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.max_samples is not None and printed >= int(args.max_samples):
                print(f"\nReached --max-samples={args.max_samples}, stopping.", file=sys.stderr)
                break

        if eval_summary:                                                                               
              for line_no, latency, metric_dict in eval_summary:                                         
                  summary_rows.append(                                                                   
                      {                                                                                  
                          "prompt_lineno": line_no,                                                      
                          "latency_ms": round(latency, 2),                                               
                          "exact_match": (metric_dict.get("exact_match")),                                 
                          "bleu": (metric_dict.get("bleu")),                                               
                          "embedding_cosine": (metric_dict.get("embedding_cosine")),                       
                          "judge_score": (metric_dict.get("judge_score")),                                 
                      }                                                                                  
                  )                                                                                                                                                            
        if summary_rows:                                                                               
            model_safe = args.model.replace("/", "_")                                                  
            csv_filename = f"{output_path.stem}.{model_safe}.metrics.csv"                              
            csv_path = output_path.with_name(csv_filename)                                             
                                                                                                        
            df = pd.DataFrame(summary_rows)                                                            
            df.to_csv(csv_path, index=False)                                                        
            print(f"Metrics table saved to {csv_path}")                                                
                                                                                                        
            print("\nEvaluation summary:")                                                             
            print(tabulate(df, headers="keys", tablefmt="github", showindex=False))                    
                                                                                                        
            if args.judge_raw and judge_explanations:                                                  
                print("\nJudge explanations:")                                                         
                print("-" * 100)                                                                       
                for line_no, raw in judge_explanations:                                                
                    print(f"Prompt {line_no}: {raw}")                                                  
                    print("-" * 100)
            
        # print("\n")
        # if printed == 0:
        #     print("No prompts were printed!", file=sys.stderr)
        #     print(f"{skipped} lines were skipped.", file=sys.stderr)
        #     return 1
        # else:
        #     print(f"{printed} prompts were printed.", file=sys.stderr)
        #     print(f"{skipped} lines were skipped.", file=sys.stderr)
        
        # print(f"{success} prompts were successful.")
        # print(f"{failed} prompts failed.")

    finally:                                                                                              
          if out_f is not None:                                                                             
              out_f.close()
                                      
if __name__ == "__main__":
    raise SystemExit(main())