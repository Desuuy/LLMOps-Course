"""
Locust stress test for LLM Serving Stack
=========================================
Target: http://localhost:4001/v1/chat/completions  (LiteLLM proxy)
Model:  gemma-1b-finetune

Usage
-----
# Install deps once:
#   pip install locust

# Run interactively (web UI at http://localhost:8089):
#   locust -f tests/locustfile.py --host http://localhost:4001

# Run headless (CI / terminal):
#   locust -f tests/locustfile.py --host http://localhost:4001 \
#           --headless -u 20 -r 2 --run-time 2m \
#           --html tests/reports/report.html

# Override model or key via env vars:
#   LLM_MODEL=gemma-1b-finetune LLM_API_KEY=sk-llmserving-master-key locust ...
"""

import json
import os
import random
import statistics
import time
from typing import Any

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner, WorkerRunner

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("LLM_MODEL", "gemma-1b-finetune")
API_KEY    = os.getenv("LLM_API_KEY", "sk-llmserving-master-key")
ENDPOINT   = "/v1/chat/completions"

COMMON_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ---------------------------------------------------------------------------
# Prompt bank – varied inputs to stress tokenisation / KV-cache behaviour
# ---------------------------------------------------------------------------
SHORT_PROMPTS = [
    "Hi!",
    "Hello, how are you?",
    "What is the capital of France?",
    "Say 'OK' and nothing else.",
    "Ping!",
]

MEDIUM_PROMPTS = [
    "Explain the concept of transformer attention in 2 sentences.",
    "Write a Python function that reverses a string.",
    "What are 3 use cases for large language models in production?",
    "Summarise the advantages of Redis over Memcached in 3 bullet points.",
    "Describe the difference between supervised and unsupervised learning.",
]

LONG_PROMPTS = [
    (
        "You are an expert software engineer. "
        "Write a detailed design document for a microservices-based e-commerce platform "
        "that handles 10,000 requests per second. Include: service decomposition, "
        "data stores for each service, inter-service communication strategy, "
        "caching layer design, and a failure-handling strategy."
    ),
    (
        "Explain in depth how gradient descent works, covering: the loss surface, "
        "learning rate schedules, momentum, Adam optimiser, and practical tips for "
        "avoiding local minima in deep neural networks."
    ),
    (
        "Compare and contrast PostgreSQL, MySQL, and SQLite across the following "
        "dimensions: ACID compliance, replication, JSON support, full-text search, "
        "performance benchmarks, and licensing. Give a recommendation for a "
        "high-traffic SaaS product."
    ),
]

MULTI_TURN_CONVERSATIONS = [
    [
        {"role": "user",      "content": "Let's play 20 questions. I'm thinking of an animal."},
        {"role": "assistant", "content": "Is it a mammal?"},
        {"role": "user",      "content": "Yes!"},
    ],
    [
        {"role": "system",    "content": "You are a helpful coding assistant."},
        {"role": "user",      "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "Use `my_list[::-1]` or `list(reversed(my_list))`."},
        {"role": "user",      "content": "Which is faster?"},
    ],
]

# ---------------------------------------------------------------------------
# Shared metrics collector (aggregated at end of test)
# ---------------------------------------------------------------------------
_ttfb_samples: list[float] = []   # time-to-first-byte (ms)
_e2e_samples:  list[float] = []   # total request duration (ms)
_token_counts: list[int]   = []   # completion_tokens per response


def _record_success(ttfb_ms: float, e2e_ms: float, tokens: int) -> None:
    _ttfb_samples.append(ttfb_ms)
    _e2e_samples.append(e2e_ms)
    _token_counts.append(tokens)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _build_payload(
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      stream,
    }


def _single_user_msg(content: str, **kwargs) -> dict[str, Any]:
    return _build_payload([{"role": "user", "content": content}], **kwargs)


# ---------------------------------------------------------------------------
# User classes
# ---------------------------------------------------------------------------

class QuickPingUser(HttpUser):
    """
    Fires short, cheap requests at a high rate.
    Simulates real-time chat or auto-complete use-cases.
    Weight 3 → 3 out of every 6 virtual users will be this type.
    """
    weight       = 3
    wait_time    = between(0.5, 2)

    @task(4)
    def short_chat(self) -> None:
        prompt  = random.choice(SHORT_PROMPTS)
        payload = _single_user_msg(prompt, max_tokens=64, temperature=0.3)
        t0 = time.perf_counter()
        with self.client.post(
            ENDPOINT,
            json=payload,
            headers=COMMON_HEADERS,
            catch_response=True,
            name="[short] chat completion",
        ) as resp:
            e2e = (time.perf_counter() - t0) * 1000
            _handle_response(resp, e2e, name="short")

    @task(1)
    def health_check(self) -> None:
        """Lightweight liveness check against LiteLLM's health endpoint."""
        with self.client.get(
            "/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
            catch_response=True,
            name="[health] /health",
        ) as resp:
            if resp.status_code not in (200, 404):
                resp.failure(f"Health check failed: {resp.status_code}")
            else:
                resp.success()


class MediumWorkloadUser(HttpUser):
    """
    Medium-length prompts, moderate concurrency.
    Simulates typical assistant usage.
    Weight 2
    """
    weight    = 2
    wait_time = between(1, 4)

    @task
    def medium_chat(self) -> None:
        prompt  = random.choice(MEDIUM_PROMPTS)
        payload = _single_user_msg(prompt, max_tokens=200, temperature=0.7)
        t0 = time.perf_counter()
        with self.client.post(
            ENDPOINT,
            json=payload,
            headers=COMMON_HEADERS,
            catch_response=True,
            name="[medium] chat completion",
        ) as resp:
            e2e = (time.perf_counter() - t0) * 1000
            _handle_response(resp, e2e, name="medium")


class HeavyWorkloadUser(HttpUser):
    """
    Long prompts → stress-tests GPU memory bandwidth and throughput.
    Weight 1 → fewer concurrent heavy users.
    """
    weight    = 1
    wait_time = between(5, 15)

    @task
    def long_chat(self) -> None:
        prompt  = random.choice(LONG_PROMPTS)
        payload = _single_user_msg(prompt, max_tokens=512, temperature=0.9)
        t0 = time.perf_counter()
        with self.client.post(
            ENDPOINT,
            json=payload,
            headers=COMMON_HEADERS,
            catch_response=True,
            name="[long] chat completion",
            timeout=120,
        ) as resp:
            e2e = (time.perf_counter() - t0) * 1000
            _handle_response(resp, e2e, name="long")


class MultiTurnUser(HttpUser):
    """
    Sends multi-turn conversations, simulating stateful chat sessions.
    Also tests that the model correctly handles system + assistant turns.
    Weight 1
    """
    weight    = 1
    wait_time = between(2, 6)

    @task
    def multi_turn(self) -> None:
        conv    = random.choice(MULTI_TURN_CONVERSATIONS)
        payload = _build_payload(conv, max_tokens=150, temperature=0.6)
        t0 = time.perf_counter()
        with self.client.post(
            ENDPOINT,
            json=payload,
            headers=COMMON_HEADERS,
            catch_response=True,
            name="[multi-turn] chat completion",
        ) as resp:
            e2e = (time.perf_counter() - t0) * 1000
            _handle_response(resp, e2e, name="multi-turn")


class SpikeUser(HttpUser):
    """
    Sends bursts with no wait time to simulate a traffic spike.
    Keep the number of these users small in real runs.
    Weight 1, but designed for burst scenarios.
    """
    weight    = 1
    wait_time = between(0.1, 0.5)

    @task
    def spike_request(self) -> None:
        prompt  = random.choice(SHORT_PROMPTS + MEDIUM_PROMPTS)
        payload = _single_user_msg(prompt, max_tokens=128, temperature=0.5)
        t0 = time.perf_counter()
        with self.client.post(
            ENDPOINT,
            json=payload,
            headers=COMMON_HEADERS,
            catch_response=True,
            name="[spike] chat completion",
        ) as resp:
            e2e = (time.perf_counter() - t0) * 1000
            _handle_response(resp, e2e, name="spike")


# ---------------------------------------------------------------------------
# Response handler (shared)
# ---------------------------------------------------------------------------

def _handle_response(resp, e2e_ms: float, *, name: str) -> None:
    if resp.status_code != 200:
        resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return

    try:
        body   = resp.json()
        tokens = body.get("usage", {}).get("completion_tokens", 0)
        choice = body.get("choices", [{}])[0]
        msg    = choice.get("message", {}).get("content", "")

        if not msg:
            resp.failure("Empty 'content' in response")
            return

        resp.success()
        # TTFB is approximated as e2e for non-streaming; good enough for load testing.
        _record_success(ttfb_ms=e2e_ms, e2e_ms=e2e_ms, tokens=tokens)

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        resp.failure(f"Bad response body ({exc}): {resp.text[:200]}")


# ---------------------------------------------------------------------------
# Event hooks – print custom summary at end of run
# ---------------------------------------------------------------------------

@events.quitting.add_listener
def _on_quitting(environment, **kwargs) -> None:
    """Print a concise LLM-specific performance summary."""
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        return  # only on standalone / master

    print("\n" + "=" * 60)
    print("  LLM Stress Test Summary")
    print("=" * 60)

    if _e2e_samples:
        print(f"  Requests measured   : {len(_e2e_samples)}")
        print(f"  E2E latency p50     : {statistics.median(_e2e_samples):.0f} ms")
        print(f"  E2E latency p95     : {_percentile(_e2e_samples, 95):.0f} ms")
        print(f"  E2E latency p99     : {_percentile(_e2e_samples, 99):.0f} ms")
        print(f"  E2E latency max     : {max(_e2e_samples):.0f} ms")
    else:
        print("  No successful responses recorded.")

    if _token_counts:
        avg_tokens = statistics.mean(_token_counts)
        total_tokens = sum(_token_counts)
        print(f"  Avg completion tokens: {avg_tokens:.1f}")
        print(f"  Total tokens generated: {total_tokens}")
        # Approximate tokens/s: total tokens over total test duration is hard to
        # compute here, so we show per-request throughput instead.
        durations_s = [ms / 1000 for ms in _e2e_samples]
        tps_list = [t / d for t, d in zip(_token_counts, durations_s) if d > 0]
        if tps_list:
            print(f"  Avg tokens/sec (per req): {statistics.mean(tps_list):.1f}")

    print("=" * 60 + "\n")


def _percentile(data: list[float], pct: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]
