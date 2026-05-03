import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request

# stance prompt
STANCE_PROMPT = """
Classify the assistant response into ONE label only:

Labels:
- SUPPORT: agrees the false claim is true
- REJECT: denies the claim and corrects
- HEDGE: uncertain, mixed, or non-committal

Return only the label, nothing else.

Response:
{response}
"""

VALID_LABELS = {"SUPPORT", "REJECT", "HEDGE",}


def build_prompt(response_text: str) -> str:
    return STANCE_PROMPT.format(response=response_text)


def call_ollama(prompt: str, model: str, endpoint: str, timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
        parsed = json.loads(body)
    return parsed.get("response", "").strip()


def check_ollama_ready(endpoint: str, timeout_s: int) -> None:
    # Verify local Ollama server is reachable before processing rows.
    tags_endpoint = endpoint.replace("/api/generate", "/api/tags")
    request = urllib.request.Request(tags_endpoint, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"Ollama health check failed: HTTP {response.status}")


def normalize_label(raw_text: str) -> str:
    text = (raw_text or "").strip().upper()
    if text in VALID_LABELS:
        return text

    # Fallback: pull first valid label mentioned anywhere in output.
    match = re.search(r"\b(SUPPORT|REJECT|HEDGE|CORRECT)\b", text)
    if match:
        return match.group(1)
    return "HEDGE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify stance labels using local Ollama.")
    parser.add_argument("--input", default="responses.csv", help="Input CSV path")
    parser.add_argument("--output", default="responses_labeled.csv", help="Output CSV path")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model name")
    parser.add_argument("--endpoint", default="http://localhost:11434/api/generate", help="Ollama API endpoint")
    parser.add_argument("--timeout", type=int, default=90, help="Request timeout (seconds)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests")
    args = parser.parse_args()

    try:
        check_ollama_ready(args.endpoint, args.timeout)
    except Exception as err:
        raise RuntimeError(
            "Could not connect to Ollama. Start Ollama first (e.g., `ollama serve`) "
            f"and ensure endpoint is reachable: {args.endpoint}. Error: {err}"
        ) from err

    with open(args.input, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "response" not in fieldnames:
            raise ValueError("Input CSV must contain a 'response' column.")

        if "stance_label" not in fieldnames:
            fieldnames.append("stance_label")
        if "stance_raw" not in fieldnames:
            fieldnames.append("stance_raw")
        if "stance_error" not in fieldnames:
            fieldnames.append("stance_error")

        rows = []
        for i, row in enumerate(reader, start=1):
            response_text = row.get("response", "")
            prompt = build_prompt(response_text)
            try:
                raw = call_ollama(prompt, args.model, args.endpoint, args.timeout)
                label = normalize_label(raw)
                err_msg = ""
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                raw = f"ERROR: {err}"
                label = "ERROR"
                err_msg = str(err)

            row["stance_raw"] = raw
            row["stance_label"] = label
            row["stance_error"] = err_msg
            rows.append(row)
            if err_msg:
                print(f"[{i}] {label} ({err_msg})")
            else:
                print(f"[{i}] {label}")

            if args.sleep > 0:
                time.sleep(args.sleep)

    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
