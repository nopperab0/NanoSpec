import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.dual_model import DualModelRunner
from nanovllm.engine.speculative import SpeculativeDecoder
from nanovllm.engine.step_timer import StepTimer


def _draft_config(config: Config, config_kwargs: dict) -> Config:
    draft_kwargs = {**config_kwargs, "kv_cache_memory_bytes": config.draft_kv_cache_memory_bytes,
                    "draft_model": None, "num_speculative_tokens": 0, "draft_kv_cache_memory_bytes": None,
                    "method": "none", "draft_type": "model"}
    draft_config = Config(config.draft_model, **draft_kwargs)
    if draft_config.max_model_len != config.max_model_len:
        raise ValueError("target and draft disagree on max_model_len; pass a max_model_len both support")
    if draft_config.hf_config.vocab_size != config.hf_config.vocab_size:
        raise ValueError("target and draft must share the vocabulary")
    return draft_config


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        unknown = set(kwargs) - config_fields
        if unknown:
            raise TypeError(f"Unknown LLM options: {sorted(unknown)}")
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.ps = []
        self.events = []
        self.dual_runner = None
        self.speculative = None
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if config.method == "naive":
            draft_tokenizer = AutoTokenizer.from_pretrained(config.draft_model, use_fast=True)
            if (draft_tokenizer.get_vocab() != self.tokenizer.get_vocab() or
                    draft_tokenizer.all_special_ids != self.tokenizer.all_special_ids):
                raise ValueError("naive draft must use the target's token-to-id mapping and special tokens")
        draft_config = None
        if config.draft_model is not None and config.draft_type == "model":
            draft_config = _draft_config(config, config_kwargs)
            self.dual_runner = DualModelRunner(config, draft_config)
            self.model_runner = self.dual_runner.target
        else:
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = ModelRunner(config, 0, self.events)
        config.eos = self.tokenizer.eos_token_id
        if config.draft_model is not None:
            if draft_config is not None:
                draft_eos = AutoTokenizer.from_pretrained(config.draft_model, use_fast=True).eos_token_id
                draft_config.eos = config.eos
            else:    # a DFlash draft directory has its config and weights, no tokenizer
                draft_eos = AutoConfig.from_pretrained(config.draft_model).eos_token_id
            if draft_eos != config.eos:
                raise ValueError("target and draft tokenizers disagree on the EOS token")
            draft_runner = self.dual_runner.draft if self.dual_runner is not None else None
            if config.method == "jetspec":
                from nanovllm.engine.jetspec import JetSpecDecoder
                self.speculative = JetSpecDecoder(self.model_runner, config)
            else:
                self.speculative = SpeculativeDecoder(self.model_runner, draft_runner, config)
            if not config.enforce_eager:
                self.speculative.capture_graphs()
            if config.record_step_timings:
                self.speculative.enable_phase_timings()
        self.scheduler = Scheduler(config, draft_config)
        self.step_timer = StepTimer(config.record_step_timings)
        self.num_steps = 0
        self.last_step = None
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        if self.speculative is not None:
            self.speculative.close()
        if self.dual_runner is not None:
            self.dual_runner.exit()
        else:
            self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> Sequence:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        budget = len(prompt) + sampling_params.max_tokens + self.config.num_speculative_tokens
        if budget > self.config.max_model_len:
            raise ValueError(f"prompt ({len(prompt)}) + max_tokens + num_speculative_tokens exceeds max_model_len ({self.config.max_model_len})")
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def start_measurement(self) -> None:
        """Forget earlier step timings, round phases and acceptance counts, and restart the step count."""
        self.step_timer.reset()
        if self.speculative is not None:
            self.speculative.reset_phase_timings()
            self.speculative.stats.clear()
        self.num_steps = 0

    def step(self):
        schedule_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        schedule_seconds = perf_counter() - schedule_start
        target_blocks = len(self.scheduler.block_manager.used_block_ids)
        draft_manager = self.scheduler.draft_block_manager
        draft_blocks = len(draft_manager.used_block_ids) if draft_manager is not None else 0
        before = [seq.num_completion_tokens for seq in seqs]
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        start_event = self.step_timer.begin()
        if is_prefill:
            if self.speculative is None:
                token_ids = self.model_runner.call("run", seqs, True)
            else:
                aux = self.speculative.prefill_aux(seqs)
                token_ids = self.model_runner.call("run", seqs, True, aux)
                self.speculative.prefill_draft(seqs, aux)
            self.scheduler.postprocess(seqs, token_ids, True)
            kind = "prefill"
        elif self.speculative is not None:
            appended = self.speculative.run_round(seqs)
            self.scheduler.postprocess_speculative(seqs, appended)
            kind = "round"
        else:
            token_ids = self.model_runner.call("run", seqs, False)
            self.scheduler.postprocess(seqs, token_ids, False)
            kind = "decode"
        delivered = [seq.num_completion_tokens - count for seq, count in zip(seqs, before)]
        if not is_prefill:
            num_tokens = -sum(delivered)
        self.step_timer.end(start_event, kind, len(seqs), sum(delivered))
        if getattr(self.config, "record_request_events", False):
            self.last_step = dict(kind=kind, schedule_seconds=schedule_seconds,
                target_blocks=target_blocks, draft_blocks=draft_blocks,
                deliveries=[dict(seq_id=seq.seq_id, tokens=count) for seq, count in zip(seqs, delivered) if count],
                scheduled_ids=[seq.seq_id for seq in seqs])
        for seq in seqs:
            if seq.first_token_step < 0 and seq.num_completion_tokens > 0:
                seq.first_token_step = self.num_steps
            if seq.is_finished:
                seq.finish_step = self.num_steps
        self.num_steps += 1
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.first_token_step, seq.finish_step)
                   for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def step_timings(self) -> list[dict]:
        """Per-step timings of the last generate() call (empty unless record_step_timings)."""
        return self.step_timer.timings()

    def phase_timings(self) -> list[dict]:
        """Per-phase timings of every speculative round of the last generate() call."""
        return self.speculative.phase_timings() if self.speculative is not None else []

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("one SamplingParams is required per prompt")
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        self.start_measurement()
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids, first_token_step, finish_step in output:
                outputs[seq_id] = (token_ids, first_token_step, finish_step)
                pbar.update(1)
        pbar.close()
        return [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids,
             "first_token_step": first_token_step, "finish_step": finish_step}
            for token_ids, first_token_step, finish_step in (outputs[seq_id] for seq_id in sorted(outputs))
        ]
