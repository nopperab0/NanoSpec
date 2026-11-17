"""Independent JetSpec: causal parallel head, candidate tree, tree verify, KV gather.

Teaching path: eager SDPA tree attention; one tree per request. It keeps nano's
ordinary FlashAttention decode untouched. This is not the optimized JetSpec engine.
"""
import torch
from transformers import AutoConfig

from nanovllm.engine.batch_metadata import prefill_metadata
from nanovllm.engine.speculative import SpeculativeDecoder
from nanovllm.engine.tree import build_tree, traverse
from nanovllm.layers.spec_sampler import sampling_tensors, probs_from_logits, sample
from nanovllm.layers.tree_attention import TreeAttention
from nanovllm.models.jetspec import JetSpecHead
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model


class JetSpecDecoder(SpeculativeDecoder):
    # Reuse tensor staging, counters and phase timers, but replace the whole round.
    def __init__(self, target, config):
        from collections import Counter
        self.target, self.config, self.device = target, config, target.device
        self.block_size = config.kvcache_block_size
        self.graphs, self.graph_vars, self.stats = {}, {}, Counter()
        self._phase_events, self._new_event, self._synchronize = [], None, None
        hf = AutoConfig.from_pretrained(config.draft_model)
        dtype, device = torch.get_default_dtype(), torch.get_default_device()
        try:
            torch.set_default_dtype(config.hf_config.dtype)
            torch.set_default_device(self.device)
            self.head = JetSpecHead(hf)
            load_model(self.head, config.draft_model)
            self.kv_cache = torch.zeros(2, hf.num_hidden_layers, config.num_kvcache_blocks + 1,
                self.block_size, hf.num_key_value_heads, hf.head_dim)
        finally:
            torch.set_default_dtype(dtype)
            torch.set_default_device(device)
        for i, layer in enumerate(self.head.layers):
            layer.self_attn.attn.k_cache = self.kv_cache[0, i]
            layer.self_attn.attn.v_cache = self.kv_cache[1, i]
        target.model.model.aux_layer_ids = self.head.aux_layer_ids
        self.aux_width = len(self.head.aux_layer_ids) * config.hf_config.hidden_size

    def prefill_aux(self, seqs):
        return torch.empty(sum(s.num_scheduled_tokens for s in seqs), self.aux_width,
                           device=self.device, dtype=self.config.hf_config.dtype)

    @torch.inference_mode()
    def prefill_draft(self, seqs, aux):
        meta = prefill_metadata(seqs, "target", self.block_size)
        self.head.store_context(aux, self._device_tensor(meta.positions, torch.int64),
                                self._device_tensor(meta.slot_mapping, torch.int32))

    def _slots(self, seq, positions):
        return [seq.block_table[p // self.block_size] * self.block_size + p % self.block_size for p in positions]

    def _propose(self, seq):
        prefix, block = len(seq) - 1, self.head.block_size
        positions = self._device_tensor(list(range(prefix, prefix + block)), torch.int64)
        ids = self._device_tensor([seq.last_token] + [self.head.mask_token_id] * (block - 1), torch.int64)
        set_context(False,
            slot_mapping=self._device_tensor(self._slots(seq, range(prefix, prefix + block)), torch.int32),
            context_lens=self._device_tensor([prefix + block], torch.int32),
            block_tables=self._device_tensor([seq.block_table], torch.int32), query_len_per_seq=block, causal=True)
        try:
            hidden = self.head(self.target.model.model.embed_tokens(ids), positions)
            selected = hidden[:block-1] if self.config.jetspec_draft_shift else hidden[1:]
            logits = self.target.model.compute_logits(selected, all_logits=True)
        finally:
            reset_context()
        return build_tree(seq.last_token, logits, self.config.jetspec_tree_width, self.config.jetspec_tree_budget)

    @torch.inference_mode()
    def run_round(self, seqs):
        appended = []
        for seq in seqs:
            assert seq.num_cached_tokens == len(seq) - 1
            mark = self._mark("start", None)
            tree = self._propose(seq)
            mark = self._mark("propose_tree", mark)
            prefix, n = len(seq) - 1, len(tree.tokens)
            scratch = self._slots(seq, range(prefix, prefix + n))
            all_slots = self._slots(seq, range(prefix)) + scratch
            meta = TreeAttention(self._device_tensor(all_slots, torch.int64), prefix, tree.ancestor_mask(self.device))
            aux = torch.empty(n, self.aux_width, device=self.device, dtype=self.config.hf_config.dtype)
            set_context(False, slot_mapping=self._device_tensor(scratch, torch.int32), tree=meta)
            try:
                hidden = self.target.model(self._device_tensor(tree.tokens, torch.int64),
                    self._device_tensor([prefix + d for d in tree.depths], torch.int64), aux)
                logits = self.target.model.compute_logits(hidden, all_logits=True)
            finally:
                reset_context()
            mark = self._mark("verify_tree", mark)
            sampling = sampling_tensors([seq], self.device)
            if sampling.temperatures is None:
                tokens = logits.argmax(-1).tolist()
                path, output = traverse(tree, lambda node: tokens[node])
            else:
                probs = probs_from_logits(logits.unsqueeze(0), sampling).squeeze(0)
                path, output = traverse(tree, lambda node: sample(probs[node], self.target.generator).item())
            mark = self._mark("accept_tree", mark)
            # Gather before writing: branch scratch and destination overlap.
            source = self._device_tensor([scratch[i] for i in path], torch.int64)
            dest = self._device_tensor(self._slots(seq, range(prefix, prefix + len(path))), torch.int64)
            flat = self.target.kv_cache.flatten(2, 3)  # [2, layers, physical slots, heads, dim]
            flat[:, :, dest] = flat[:, :, source].clone()
            self.head.store_context(aux[self._device_tensor(path, torch.int64)],
                self._device_tensor(list(range(prefix, prefix + len(path))), torch.int64), dest.to(torch.int32))
            self._mark("commit_tree", mark)
            self.stats[f"accepted_{len(path)-1}"] += 1
            self.stats["proposed_nodes"] += n - 1
            self.stats["verified_positions"] += n
            appended.append(output)
        self.stats["rounds"] += 1
        return appended

    def close(self):
        # Release head KV and weights before the target process group is closed.
        self.graphs.clear()
        self.graph_vars.clear()
        del self.head, self.kv_cache
