from __future__ import annotations
from typing import Dict, Any
import os, time, requests
from dotenv import load_dotenv

load_dotenv()

ROUTER_V1 = "https://router.huggingface.co/v1" #openai compatible
ROUTER_TASK = "https://router.huggingface.co/hf-inference/models" #legacy hf inference endpoint

def _get_hf_token() -> str:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACEHUB_API_KEY"):
        v = os.getenv(name)
        if v:
            return v
    raise RuntimeError("HF token not found.")

def _post(url: str, headers: Dict[str,str], payload: Dict[str,Any], timeout: float):
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    return r

def call_model(
    *,
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    retries: int = 3,
    timeout: float = 60.0,
    use_v1: bool = True,
) -> str:
    token = _get_hf_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    backoff = 1.0
    last_err = None

    for _ in range(max(1, retries)):
        try:
            if use_v1:
                # openai style
                url = f"{ROUTER_V1}/chat/completions"
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(max_tokens),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    #"stream": True,
                }
            else:
                # Legacy
                url = f"{ROUTER_TASK}/{model}"
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": int(max_tokens),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "repetition_penalty": float(repetition_penalty),
                        "return_full_text": False,
                    },
                    "options": {"wait_for_model": True},
                }

            r = _post(url, headers, payload, timeout)
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff*2, 16.0)
            continue

        if r.status_code in (502, 503, 524, 408, 429):
            time.sleep(backoff)
            backoff = min(backoff*2, 16.0)
            continue

        if r.ok:
            data = r.json()
            if use_v1:
                try:
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    raise RuntimeError(f"Unexpected /v1 response: {data}")
            else:
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return str(data[0]["generated_text"])
                if isinstance(data, dict) and "generated_text" in data:
                    return str(data["generated_text"])
                if isinstance(data, dict) and "error" in data:
                    raise RuntimeError(f"HuggingFace API error: {data['error']}")
                raise RuntimeError(f"Unexpected task response: {data}")

        if r.status_code == 404:
            if use_v1:
                raise RuntimeError(
                    "404 Not Found from Router /v1, try using use_v1=False"
                )
            else:
                raise RuntimeError(
                    "404 Not Found from hf-inference task endpoint, try using use_v1=True"
                )
        raise RuntimeError(f"HuggingFace API error: {r.status_code}: {r.text}")

    if last_err:
        raise RuntimeError(f"HuggingFace API request failed: {last_err}")
    raise RuntimeError("HuggingFace API retries exceeded maximum amount.")


