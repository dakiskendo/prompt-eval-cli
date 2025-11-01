import os, requests
from dotenv import load_dotenv
load_dotenv()

ROUTER_V1 = "https://router.huggingface.co/v1"

def list_router_models(provider_suffix: str | None = "hf-inference") -> list[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_KEY")
    if not token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN).")
    r = requests.get(f"{ROUTER_V1}/models", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    models = [m["id"] for m in data.get("data", []) if isinstance(m, dict) and "id" in m]
    if provider_suffix:
        models = [m for m in models if m.endswith(":" + provider_suffix)]
    return models