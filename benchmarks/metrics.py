"""Host-observed request latency and explicitly bounded throughput windows."""
import math


def distribution(values):
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, max=None)
    values = sorted(values)
    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return dict(count=len(values), mean=sum(values) / len(values), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), max=values[-1])


def summarize(requests, events, wall_seconds, slo):
    rows = []
    for request in requests:
        deliveries = request["deliveries"]
        first = deliveries[0]["time_s"]
        finish = deliveries[-1]["time_s"]
        tokens = sum(d["tokens"] for d in deliveries)
        gaps = [b["time_s"] - a["time_s"] for a, b in zip(deliveries, deliveries[1:])]
        ttft = first - request["arrival_s"]
        tpot = (finish - first) / (tokens - 1) if tokens > 1 else None
        passed = ttft * 1000 <= slo["ttft_ms"]
        passed &= tpot is None or tpot * 1000 <= slo["tpot_ms"]
        passed &= not gaps or max(gaps) * 1000 <= slo["gap_ms"]
        rows.append(dict(index=request["index"], arrival_s=request["arrival_s"],
            ttft_ms=ttft * 1000, latency_ms=(finish - request["arrival_s"]) * 1000,
            queue_ms=(request["first_scheduled_s"] - request["arrival_s"]) * 1000,
            tpot_ms=tpot * 1000 if tpot is not None else None,
            delivery_gaps_ms=[g * 1000 for g in gaps],
            delivery_sizes=[d["tokens"] for d in deliveries], output_tokens=tokens, slo_passed=bool(passed)))
    full = [e for e in events if e["steady"]]
    steady_seconds = full[-1]["end_s"] - full[0]["start_s"] if full else 0.0
    steady_tokens = sum(e["tokens"] for e in full)
    decode = [e for e in events if e["kind"] != "prefill"]
    decode_seconds = sum(e["end_s"] - e["start_s"] for e in decode)
    committed = sum(e["tokens"] for e in decode)
    round_rows = sum(e["delivering_requests"] for e in decode)
    return dict(requests=rows, request_count=len(rows), requests_s=len(rows) / wall_seconds,
        latency_ms=distribution([r["latency_ms"] for r in rows]),
        ttft_ms=distribution([r["ttft_ms"] for r in rows]),
        queue_ms=distribution([r["queue_ms"] for r in rows]),
        tpot_ms=distribution([r["tpot_ms"] for r in rows if r["tpot_ms"] is not None]),
        delivery_gap_ms=distribution([gap for r in rows for gap in r["delivery_gaps_ms"]]),
        delivery_tokens=distribution([size for r in rows for size in r["delivery_sizes"]]),
        steady_seconds=steady_seconds, steady_tokens=steady_tokens,
        steady_tok_s=steady_tokens / steady_seconds if steady_seconds else None,
        decode_step_tok_s=committed / decode_seconds if decode_seconds else None,
        committed_decode_tokens=committed,
        committed_tokens_per_sequence_round=committed / round_rows if round_rows else None,
        slo=slo, goodput_requests_s=sum(r["slo_passed"] for r in rows) / wall_seconds,
        goodput_tokens_s=sum(r["output_tokens"] for r in rows if r["slo_passed"]) / wall_seconds)
