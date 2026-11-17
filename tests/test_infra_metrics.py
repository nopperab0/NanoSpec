from types import SimpleNamespace

import pytest

from benchmarks.metrics import distribution, summarize
from benchmarks.runner import run_requests


def test_burst_latency_and_slo_use_host_delivery_not_fake_token_times():
    requests = [dict(index=0, arrival_s=0, first_scheduled_s=.02,
        deliveries=[dict(time_s=.1, tokens=1), dict(time_s=.2, tokens=3), dict(time_s=.3, tokens=2)])]
    result = summarize(requests, [], .4, dict(ttft_ms=150, tpot_ms=50, gap_ms=150))
    assert result["ttft_ms"]["p50"] == 100
    assert result["tpot_ms"]["p50"] == pytest.approx(40)
    assert result["delivery_gap_ms"]["count"] == 2
    assert result["delivery_gap_ms"]["p99"] == pytest.approx(100)
    assert result["goodput_requests_s"] == 2.5
    assert result["steady_tok_s"] is None
    assert result["requests"][0]["queue_ms"] == 20


def test_distribution_empty_and_interpolated_quantiles():
    assert distribution([])["p99"] is None
    assert distribution([1, 3])["p50"] == 2
    assert distribution([1, 3])["p99"] == pytest.approx(2.98)


class Clock:
    time = 0.0
    def __call__(self):
        return self.time
    def wait(self, seconds):
        self.time += seconds


class Engine:
    def __init__(self, clock, concurrency):
        self.clock = clock
        self.concurrency = concurrency
        self.next_id = 0
        self.scheduler = SimpleNamespace(waiting=[], running=[], preemptions=0, recomputed_tokens=0,
            block_manager=SimpleNamespace(used_block_ids=set(), blocks=list(range(20))), draft_block_manager=None)

    def start_measurement(self):
        pass

    def add_request(self, prompt, params):
        seq = SimpleNamespace(seq_id=self.next_id, generated=0, maximum=params.max_tokens)
        self.next_id += 1
        self.scheduler.waiting.append(seq)
        return seq

    def step(self):
        scheduler = self.scheduler
        while scheduler.waiting and len(scheduler.running) < self.concurrency:
            scheduler.running.append(scheduler.waiting.pop(0))
        self.clock.time += .01
        deliveries, finished, ids = [], [], []
        prefill = any(seq.generated == 0 for seq in scheduler.running)
        for seq in list(scheduler.running):
            count = min(1 if seq.generated == 0 else 2, seq.maximum - seq.generated)
            seq.generated += count
            ids.append(seq.seq_id)
            deliveries.append(dict(seq_id=seq.seq_id, tokens=count))
            if seq.generated == seq.maximum:
                finished.append((seq.seq_id, [3] * seq.maximum, 0, 1))
                scheduler.running.remove(seq)
        self.last_step = dict(kind="prefill" if prefill else "round", schedule_seconds=.001,
            deliveries=deliveries, scheduled_ids=ids, target_blocks=2 * len(ids), draft_blocks=0)
        return finished, 0


@pytest.mark.parametrize("mode", ["bulk", "closed", "rate"])
def test_request_driver_refill_queue_and_real_committed_counts(mode):
    clock = Clock()
    engine = Engine(clock, 2)
    result = run_requests(engine, [[1], [2]], SimpleNamespace(max_tokens=6),
        dict(load=mode, concurrency=2, num_requests=6, arrival_rate=10), clock=clock, wait=clock.wait)
    assert len(result["outputs"]) == 6
    assert sum(row["output_tokens"] for row in result["metrics"]["requests"]) == 36
    assert result["metrics"]["peak_target_blocks"] == (4 if mode != "rate" else 2)
    if mode == "closed":
        assert result["metrics"]["steady_tok_s"] > 0
    else:
        assert result["metrics"]["steady_tok_s"] is None
    if mode == "bulk":
        assert result["metrics"]["queue_ms"]["max"] > 0
    if mode == "rate":
        assert result["request_events"][-1]["arrival_s"] == .5
