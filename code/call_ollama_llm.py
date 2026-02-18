import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen3:8b",
    "prompt": "请用中文列出 RAG 流程步骤。",
    "max_tokens": 200,
    "stream": False,
}
resp = requests.post(url, json=payload, timeout=300)
print(resp.json())
