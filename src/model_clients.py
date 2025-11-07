from __future__ import annotations
from typing import Dict, Any, List
import os, time, requests, numpy as np
from dotenv import load_dotenv

load_dotenv()

ROUTER_V1 = "https://router.huggingface.co/v1" #openai compatible                        
FEAT_EXTRACT = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction"  # embeddings via feature-extraction

class EmbeddingNotFoundError(RuntimeError):
    """Raised when an embeddings endpoint reports a 404."""

def _parse_embedding_payload(data: Any) -> List[float]:                                                   
      """                                                                                                   
      Normalize embedding payloads coming from different Hugging Face endpoints.                            
                                                                                                            
      Router `/v1/embeddings` returns: {"data": [{"embedding": [...]}, ...], ...}                           
      Feature-extraction pipeline can return a nested list (per-token) or a flat list.                      
      """                                                                                                   
      if isinstance(data, dict):                                                                            
          if "data" in data:                                                                                
              return _parse_embedding_payload(data["data"])                                                 
          if "embedding" in data:                                                                           
              return _parse_embedding_payload(data["embedding"])                                            
          if "embeddings" in data:                                                                          
              return _parse_embedding_payload(data["embeddings"])                                           
          if "error" in data:                                                                               
              raise RuntimeError(f"HuggingFace API error: {data['error']}")                                 
                                                                                                            
      if isinstance(data, list):                                                                            
          if not data:                                                                                      
              raise RuntimeError("Empty embedding response")                                                
          if all(isinstance(x, (int, float)) for x in data):                                                
              return [float(x) for x in data]                                                               
          if len(data) == 1:                                                                                
              return _parse_embedding_payload(data[0])                                                      
          if all(isinstance(vec, list) for vec in data):                                                    
              first_len = len(data[0])                                                                      
              if first_len == 0:                                                                            
                  raise RuntimeError("Empty embedding vectors")                                             
              for vec in data:                                                                              
                  if len(vec) != first_len:                                                                 
                      raise RuntimeError("Inconsistent embedding vector lengths")                           
              sums = [0.0] * first_len                                                                      
              for vec in data:                                                                              
                  for idx, val in enumerate(vec):                                                           
                      sums[idx] += float(val)                                                               
              divisor = len(data)                                                                           
              return [val / divisor for val in sums]                                                        
                                                                                                            
      snippet = str(data)                                                                                   
      if len(snippet) > 200:                                                                                
          snippet = snippet[:200] + "..."                                                                   
      raise RuntimeError(f"Unexpected embeddings response format: {snippet}")

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
            url = f"{ROUTER_V1}/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty)
                #"stream": True,
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
            try:
                return data["choices"][0]["message"]["content"]
            except Exception:
                raise RuntimeError(f"Unexpected response: {data}")

        if r.status_code == 404:
            raise RuntimeError(
                "404 Not Found."
            )
        raise RuntimeError(f"HuggingFace API error: {r.status_code}: {r.text}")

    if last_err:
        raise RuntimeError(f"HuggingFace API request failed: {last_err}")
    raise RuntimeError("HuggingFace API retries exceeded maximum amount.")

def _ensure_sentence_transformer(model: str):                                                             
      from sentence_transformers import SentenceTransformer                                                 
                                                                                                            
      return SentenceTransformer(model)

def _flatten_embedding(data: Any) -> List[float]:
      if np is not None and isinstance(data, np.ndarray):                                                     
        if data.ndim == 1:                                                                                  
            return [float(x) for x in data]                                                                 
        if data.ndim == 2:                                                                                  
            return [float(x) for x in data.mean(axis=0)]                                                    
        raise RuntimeError(f"Unsupported embedding shape: {data.shape}")                                                     
      if isinstance(data, dict):                                                                            
          if "data" in data:                                                                                
              return _flatten_embedding(data["data"])                                                       
          if "embedding" in data:                                                                           
              return _flatten_embedding(data["embedding"])
          if "embeddings" in data:                                                                          
              return _flatten_embedding(data["embeddings"])                                                 
          if "error" in data:                                                                               
              raise RuntimeError(f"HuggingFace API error: {data['error']}")                                 
      if isinstance(data, list):                                                                            
          if not data:                                                                                      
              raise RuntimeError("Empty embedding response")                                                
          if all(isinstance(x, (int, float)) for x in data):                                                
              return [float(x) for x in data]                                                               
          if len(data) == 1:                                                                                
              return _flatten_embedding(data[0])                                                            
          if all(isinstance(row, list) for row in data):                                                    
              length = len(data[0])                                                                         
              for row in data:                                                                              
                  if len(row) != length:                                                                    
                      raise RuntimeError("Inconsistent embedding lengths")                                  
              sums = [0.0] * length                                                                         
              for row in data:                                                                              
                  for idx, val in enumerate(row):                                                           
                      sums[idx] += float(val)                                                               
              return [val / len(data) for val in sums]                                                      
      snippet = str(data)                                                                                   
      if len(snippet) > 200:                                                                                
          snippet = snippet[:200] + "..."                                                                   
      raise RuntimeError(f"Unexpected embedding payload: {snippet}")
                                                                                                        
def get_embedding(                                                                                   
      *,                                                                                                    
      text: str,                                                                                            
      model: str = "thenlper/gte-small",                                                                    
      retries: int = 3,                                                                                     
      timeout: float = 60.0,                                                                                                                                                                 
  ) -> List[float]:                                                                                         
      token = _get_hf_token()                                                                               
                                                                                                                                         
      try:                                                                                                  
          from huggingface_hub import InferenceClient                                                       
          from huggingface_hub.utils import HfHubHTTPError                                                  
                                                                                                            
          client = InferenceClient(model=model, token=token, timeout=timeout)                               
          data = client.feature_extraction(text)                                       
          return _flatten_embedding(data)                                                                   
                                                                                                            
      except ImportError as exc:                                                                            
          hf_error = exc                                                                                    
      except HfHubHTTPError as exc:                                                                         
          if exc.response is not None and exc.response.status_code == 404:                                  
              hf_error = EmbeddingNotFoundError(f"No hosted embeddings for {model}")                        
          else:                                                                                             
              raise                                                                                         
      except Exception as exc:
          raise RuntimeError(f"HuggingFace inference error: {exc}") from exc                                
      else:                                                                                                 
          hf_error = None                                                                                   
                                                                                                                                               
      try:                                                                                                  
          encoder = _ensure_sentence_transformer(model)                                                     
      except ImportError:                                                                                   
          if isinstance(hf_error, EmbeddingNotFoundError):                                                  
              raise hf_error                                                                                
          raise RuntimeError(                                                                               
              "Install sentence-transformers or pick a hosted embedding model."                             
          )                                                                                                 
      except Exception as exc:                                                                              
          raise RuntimeError(                                                                               
              f"Could not load sentence-transformers model '{model}': {exc}"                                
          )     
                                                                                                                                                                                               
      vector = encoder.encode(text, convert_to_numpy=False, normalize_embeddings=False)                     
      return [float(x) for x in vector]
