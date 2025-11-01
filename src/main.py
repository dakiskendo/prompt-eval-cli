from __future__ import annotations
import json, argparse, sys, time
from pathlib import Path
from tqdm import tqdm #unused for now
from datetime import datetime, timezone
from model_clients import call_model
from typing import Iterator, Any

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
        "--ask-model",
        action="store_true",
        help="Alternative prompt if not provided",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "outputs.jsonl",
        help="Path to write JSONL output results",
    )
    parser.add_argument(
        "--api",
        choices=["v1", "legacy"],
        default="v1",
        help="API path: Choose between Router /v1 (openai) or legacy task endpoint"
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
    if not model and args.ask_model:
        model = input("Enter HuggingFace model (id or full url): ").strip()
    if not model:
        model = "meta-llama/Meta-Llama-3-8B-Instruct" # if user doesn't provide a desired model, use this one as a default
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 2
    if not input_path.is_file():
        print(f"Error: input path is not a file: {input_path}", file=sys.stderr)
        return 2
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    printed = 0
    skipped = 0
    success = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
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

            printed += 1
            print(f"\nPrompt {printed} --> " + "'" + prompt + "'")

            started = time.perf_counter()
            output_text: str | None = None
            error_text: str | None = None
            try:
                output_text = call_model(
                    prompt=prompt,
                    model=model,
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    repetition_penalty=float(args.repetition_penalty),
                    use_v1=(args.api == "v1"),
                    timeout=60.0,
                )
                success += 1
            except Exception as e:
                error_text = str(e)
                failed += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(f"Elapsed time: {elapsed_ms:.3f} ms")

            record = {
                "lineno": lineno,
                "prompt": prompt,
                "model": model,
                "output": output_text,
                "error": error_text,
                "latency_ms": round(elapsed_ms, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if printed == 0:
            print("No prompts were printed!", file=sys.stderr)
            print(f"\n{skipped} lines were skipped.", file=sys.stderr)
            return 1
        else:
            print(f"\n{printed} prompts were printed.", file=sys.stderr)
            print(f"\n{skipped} lines were skipped.", file=sys.stderr)
        
        print(f"\n{success} prompts were successful.")
        print(f"\n{failed} prompts failed.")

if __name__ == "__main__":
    raise SystemExit(main())