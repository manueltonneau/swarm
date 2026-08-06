#!/usr/bin/env python3
"""Shared utilities for OpenAI Batch API operations."""

import json
import sys
import time

from openai import OpenAI


POLL_INTERVAL = 30  # seconds


def api_call_with_retry(fn, *args, **kwargs):
    """Call any OpenAI API function, retrying on network errors."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            wait = min(10 * attempt, 120)
            print(f"  [retry {attempt}] {type(e).__name__}: {str(e)[:80]} — waiting {wait}s",
                  file=sys.stderr)
            time.sleep(wait)


def submit_batch(client, input_path, state_path):
    """Upload JSONL and submit a batch job. Saves state for resume."""
    # Check for existing batch
    try:
        with open(state_path) as f:
            state = json.load(f)
        if "batch_id" in state:
            print(f"Resuming existing batch: {state['batch_id']}", file=sys.stderr)
            return state["batch_id"]
    except FileNotFoundError:
        pass

    # Upload
    print(f"Uploading {input_path}...", file=sys.stderr)
    with open(input_path, "rb") as f:
        input_file = api_call_with_retry(client.files.create, file=f, purpose="batch")
    print(f"  File ID: {input_file.id}", file=sys.stderr)

    # Submit
    batch = api_call_with_retry(
        client.batches.create,
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"  Batch submitted: {batch.id}", file=sys.stderr)

    # Save state
    state = {"batch_id": batch.id, "file_id": input_file.id}
    with open(state_path, "w") as f:
        json.dump(state, f)

    return batch.id


def poll_batch(client, batch_id):
    """Poll until batch completes. Returns the batch object."""
    while True:
        job = api_call_with_retry(client.batches.retrieve, batch_id)
        counts = job.request_counts
        completed = getattr(counts, "completed", 0)
        failed = getattr(counts, "failed", 0)
        total = getattr(counts, "total", 0)
        print(f"  [{time.strftime('%H:%M:%S')}] status={job.status}  "
              f"completed={completed}/{total}  failed={failed}", file=sys.stderr)
        if job.status == "completed":
            return job
        elif job.status in ("failed", "expired", "cancelled"):
            print(f"Batch ended with status: {job.status}", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_INTERVAL)


def download_results(client, job, output_path, error_path=None):
    """Download batch output (and error file if present)."""
    print(f"Downloading output (file_id={job.output_file_id})...", file=sys.stderr)
    stream = api_call_with_retry(client.files.content, job.output_file_id)
    with open(output_path, "wb") as f:
        f.write(stream.read())

    if job.error_file_id and error_path:
        print(f"Downloading errors (file_id={job.error_file_id})...", file=sys.stderr)
        err_stream = api_call_with_retry(client.files.content, job.error_file_id)
        with open(error_path, "wb") as f:
            f.write(err_stream.read())


def parse_batch_results(output_path):
    """Parse batch output JSONL. Returns dict of custom_id -> response text."""
    results = {}
    with open(output_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                cid = d["custom_id"]
                body = d["response"]["body"]

                # Find message in output array
                text = None
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") in ("output_text", "text"):
                                text = c.get("text", "").strip()
                                break
                        break

                if text:
                    results[cid] = text
                else:
                    status = body.get("status", "unknown")
                    reason = (body.get("incomplete_details") or {}).get("reason", "")
                    print(f"  No text for {cid}: status={status} reason={reason}",
                          file=sys.stderr)
                    results[cid] = None
            except Exception as e:
                cid = d.get("custom_id", "unknown")
                print(f"  Parse error for {cid}: {e}", file=sys.stderr)
                results[cid] = None
    return results


def build_request(custom_id, prompt, model="gpt-5-nano", max_output_tokens=16384,
                  effort="low"):
    """Build a single batch request record."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": effort},
        },
    }


def write_batch_input(records, path):
    """Write a list of request records as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} requests to {path}", file=sys.stderr)
