"""
End-to-end smoke test for the Stacks backend.

Run this after starting the server (`uvicorn app.main:app --reload`) to
verify every endpoint works before a demo. Uses only the Python standard
library, so it needs no extra dependencies beyond what's already installed
for the server itself.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base http://localhost:8000
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid

SAMPLE_TEXT = b"""Stacks Smoke Test Document

This is a short test document used to verify the Stacks backend pipeline.
It talks about rivers. The Amazon river is the largest river by discharge
volume in the world. The Nile river is traditionally considered the
longest river in the world, though this has been debated by geographers.

Rivers are vital for transportation, agriculture, and drinking water
supply across the world.
"""


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        sys.exit(1)


def request(method, url, data=None, is_json=True, headers=None):
    headers = headers or {}
    body = None
    if data is not None:
        if is_json:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            body = data
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def multipart_upload(url, filename, content, content_type="text/plain"):
    boundary = uuid.uuid4().hex
    lines = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        content,
        f"--{boundary}--".encode(),
        b"",
    ]
    body = b"\r\n".join(lines)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print(f"Testing Stacks backend at {base}\n")

    status, data = request("GET", base + "/health")
    check("GET /health returns 200", status == 200)
    check("GET /health reports status ok", data.get("status") == "ok")
    llm_configured = data.get("llm_configured", False)
    print(f"       llm_configured = {llm_configured} "
          f"({'answers will be LLM-generated' if llm_configured else 'retrieval-only fallback expected'})\n")

    status, data = request("GET", base + "/documents")
    check("GET /documents returns 200", status == 200)
    check("GET /documents returns a list", isinstance(data, list))

    status, data = multipart_upload(base + "/documents/upload", "smoke-test.txt", SAMPLE_TEXT)
    check("POST /documents/upload returns 200", status == 200)
    doc = data.get("document", {})
    check("Uploaded document has an id", bool(doc.get("doc_id")))
    check("Uploaded document has chunks", doc.get("chunk_count", 0) > 0)
    doc_id = doc["doc_id"]
    print(f"       indexed '{doc.get('name')}' — {doc.get('words')} words, "
          f"{doc.get('chunk_count')} chunks\n")

    status, data = request("GET", base + "/documents")
    check("Document appears in GET /documents", any(d["doc_id"] == doc_id for d in data))

    status, data = request("POST", base + "/query", {
        "question": "What is the longest river in the world?",
        "top_k": 3,
        "hybrid_rerank": True,
    })
    check("POST /query returns 200", status == 200)
    check("Query response has an answer", bool(data.get("answer")))
    check("Query response has sources", len(data.get("sources", [])) > 0)
    print(f"       answer: {data.get('answer')[:150]}...")
    print(f"       used_llm = {data.get('used_llm')}\n")

    # Follow-up question with conversation history (tests context-aware retrieval)
    status, data2 = request("POST", base + "/query", {
        "question": "and which one is it traditionally considered?",
        "top_k": 3,
        "hybrid_rerank": False,
        "history": [{"question": "What is the longest river in the world?", "answer": data.get("answer", "")}],
    })
    check("POST /query with history returns 200", status == 200)
    check("Follow-up response has sources", len(data2.get("sources", [])) > 0)
    check("Sources include a page_number field", "page_number" in data2["sources"][0])

    status, data = request("DELETE", base + f"/documents/{doc_id}")
    check("DELETE /documents/{id} returns 200", status == 200)
    check("Deletion reports removed chunks", data.get("deleted_chunks", 0) > 0)

    status, data = request("GET", base + "/documents")
    check("Document no longer listed after deletion", not any(d["doc_id"] == doc_id for d in data))

    print("\nAll checks passed — the backend is working end to end.")


if __name__ == "__main__":
    main()
