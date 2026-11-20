import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import jsonschema
import openai
import pytest

import main
from assignment import pipeline

from assignment.pipeline import (
    build_observability,
    build_production_plugins,
    run_assignment_suite,
)
from core.openai_runtime import OpenAIRunner
from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter, load_lab_pii_dataset


def test_blue_pipeline_records_actual_decisions(monkeypatch, tmp_path):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs["messages"][-1]["content"])
        text = "Visit online banking. Test contact: qa@example.com."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: client)
    plugins = build_production_plugins()
    audit, monitor = build_observability()
    monitor.rate_limit_hit_threshold = 4
    results = asyncio.run(run_assignment_suite(
        {"plugins": plugins, "audit": audit, "monitor": monitor},
        output_dir=tmp_path,
    ))

    def read(name):
        return json.loads((tmp_path / name).read_text(encoding="utf-8"))

    schema = Path(__file__).resolve().parents[2] / "schemas" / "results.schema.json"
    assert read("results.json") == results
    jsonschema.validate(results, json.loads(schema.read_text(encoding="utf-8")))
    assert all(not row["blocked"] and row["layer"] == "output_guardrail"
               for row in results["safe_queries"])
    assert all(row["blocked"] and row["input"] not in calls for row in results["attack_queries"])
    assert results["rate_limit"]["blocked"] == 5
    assert results["rate_limit"]["passed"] == 10

    entries = read("audit_log.json")
    metrics = read("metrics.json")
    assert len(entries) == metrics["total_requests"]
    assert len(calls) + metrics["blocked_requests"] + results["rate_limit"]["passed"] == len(entries)
    assert len({entry["request_id"] for entry in entries}) == len(entries)
    assert all(entry["latency_ms"] >= 0 and "qa@example.com" not in entry["output"]
               for entry in entries)
    assert any(alert["metric"] == "rate_limit_hits" for alert in metrics["alerts"])


def test_content_filter_matches_lab_pii_dataset():
    for case in load_lab_pii_dataset()["pii_cases"]:
        result = content_filter(case["input_text"])
        detected = {issue.split(":", 1)[0] for issue in result["issues"]}
        assert result["safe"] == case["expect_safe"], case["id"]
        assert detected == set(case["expect_issue_types"]), case["id"]
        assert ("[REDACTED]" in result["redacted"]) == case["expect_contains_redacted"], case["id"]


def test_banking_email_in_vietnamese_is_allowed():
    message = "Tóm tắt email về giao dịch chuyển khoản bị chậm."
    assert detect_injection(message) == "ALLOW"
    assert topic_filter("Tóm tắt email chuyển khoản delay.") == "ALLOW"


def test_cp3_reports_provider_rate_limit(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"Retry-After": "5"})

    async def fail_suite(_pipeline):
        raise openai.RateLimitError("Provider returned error", response=response, body=None)

    monkeypatch.setattr(main, "get_openrouter_api_key", lambda: "test-key")
    monkeypatch.setattr(pipeline, "run_assignment_suite", fail_suite)
    with pytest.raises(SystemExit, match=r"HTTP 429.*Wait at least 5 seconds"):
        asyncio.run(main.part3_assignment_suite())
