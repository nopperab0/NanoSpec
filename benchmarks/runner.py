"""Bulk, closed-loop, and rate-driven arrivals over nano's add_request/step API."""
from time import perf_counter, sleep

from benchmarks.metrics import summarize


def run_requests(llm, prompts, params, job, profiler=None, clock=perf_counter, wait=sleep, synchronize=lambda: None):
    mode = job.get("load", "closed")
    concurrency = job["concurrency"]
    count = job.get("num_requests", 0) or len(prompts)
    rate = job.get("arrival_rate", 1.0)
    scheduler = llm.scheduler
    scheduler.preemptions = scheduler.recomputed_tokens = 0
    llm.start_measurement()
    records, pending, outputs, events = [], {}, {}, []
    next_index = 0
    origin = clock()
    peak_target_blocks = peak_draft_blocks = 0

    def admit(arrival):
        nonlocal next_index
        seq = llm.add_request(prompts[next_index % len(prompts)], params)
        record = dict(index=next_index, seq_id=seq.seq_id, arrival_s=arrival,
                      first_scheduled_s=None, deliveries=[])
        records.append(record)
        pending[seq.seq_id] = record
        next_index += 1

    while next_index < count or pending:
        now = clock() - origin
        if mode == "bulk":
            while next_index < count:
                admit(0.0)
        elif mode == "closed":
            while next_index < count and len(pending) < concurrency:
                admit(now)
        else:
            while next_index < count and next_index / rate <= now:
                admit(next_index / rate)
        if not pending:
            wait(max(0.0, next_index / rate - (clock() - origin)))
            continue
        full = len(pending) >= concurrency
        cohort_ready = len(records) >= concurrency and all(r["deliveries"] for r in records[:concurrency])
        start = clock() - origin
        finished, _ = llm.step()
        end = clock() - origin
        step = llm.last_step
        for seq_id in step["scheduled_ids"]:
            if pending[seq_id]["first_scheduled_s"] is None:
                pending[seq_id]["first_scheduled_s"] = start
        for delivery in step["deliveries"]:
            pending[delivery["seq_id"]]["deliveries"].append(dict(time_s=end, tokens=delivery["tokens"]))
        for seq_id, token_ids, _, _ in finished:
            outputs[pending[seq_id]["index"]] = dict(token_ids=token_ids)
            del pending[seq_id]
        # Refill immediately; admission latency belongs to the measurement window.
        if mode == "closed":
            while next_index < count and len(pending) < concurrency:
                admit(clock() - origin)
        steady = mode == "closed" and cohort_ready and full and len(pending) == concurrency
        target = scheduler.block_manager
        peak_target_blocks = max(peak_target_blocks, step["target_blocks"])
        draft = scheduler.draft_block_manager
        if draft is not None:
            peak_draft_blocks = max(peak_draft_blocks, step["draft_blocks"])
        events.append(dict(start_s=start, end_s=end, kind=step["kind"], steady=steady,
            tokens=sum(d["tokens"] for d in step["deliveries"]),
            delivering_requests=len(step["deliveries"]), schedule_seconds=step["schedule_seconds"],
            waiting=len(scheduler.waiting), running=len(scheduler.running)))
        if profiler is not None:
            profiler.step()
    synchronize()
    elapsed = clock() - origin
    slo = dict(ttft_ms=job.get("slo_ttft_ms", 1000), tpot_ms=job.get("slo_tpot_ms", 50),
               gap_ms=job.get("slo_gap_ms", 100))
    metrics = summarize(records, events, elapsed, slo)
    metrics.update(preemptions=scheduler.preemptions, recomputed_tokens=scheduler.recomputed_tokens,
                   peak_target_blocks=peak_target_blocks, peak_draft_blocks=peak_draft_blocks,
                   target_block_capacity=len(target.blocks),
                   draft_block_capacity=len(draft.blocks) if draft is not None else 0,
                   scheduler_seconds=sum(e["schedule_seconds"] for e in events))
    return dict(wall_seconds=elapsed, metrics=metrics, request_events=records, host_steps=events,
                outputs=[outputs[i] for i in range(count)])
