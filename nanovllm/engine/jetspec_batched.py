"""Batched JetSpec execution, with GPU-resident trees and optional forward graphs."""
import torch

from nanovllm.layers.paged_tree_attention import PagedTreeAttention
from nanovllm.layers.spec_sampler import probs_from_logits, sample, sampling_tensors
from nanovllm.layers.tree_kernels import build_trees, commit_kv, traverse_trees
from nanovllm.utils.context import reset_context, set_context


class BatchedJetSpec:
    def __init__(self, decoder):
        self.decoder = decoder
        self.buffers = {}
        self.graphs = {}
        self.nodes = decoder.config.jetspec_tree_budget
        self.block = decoder.head.block_size
        self.path_size = min(self.block, self.nodes)
        self.table_width = (decoder.config.max_model_len + decoder.block_size - 1) // decoder.block_size

    def _allocate(self, batch):
        d, n, p, block = self.decoder, self.nodes, self.path_size, self.block
        device, dtype = d.device, d.config.hf_config.dtype
        def empty(*shape, dtype=torch.int64):
            return torch.zeros(*shape, device=device, dtype=dtype)
        cache = d.target.kv_cache
        b = dict(meta=empty(batch, self.table_width + 2), tables=empty(batch, self.table_width, dtype=torch.int32),
            prefix=empty(batch), anchors=empty(batch), active=empty(batch, dtype=torch.bool),
            head_ids=empty(batch, block), head_positions=empty(batch, block),
            head_slots=empty(batch, block, dtype=torch.int32), head_lens=empty(batch, dtype=torch.int32),
            tokens=empty(batch, n), parents=empty(batch, n, dtype=torch.int32),
            depths=empty(batch, n, dtype=torch.int32), ancestors=empty(batch, n, dtype=torch.uint64),
            positions=empty(batch, n), slots=empty(batch, n, dtype=torch.int32),
            aux=empty(batch*n, d.aux_width, dtype=dtype), path=empty(batch, p),
            packed=empty(batch, p+1), counts=empty(batch, dtype=torch.int32),
            kv_temporary=empty(2, cache.shape[1], batch*p, cache.shape[-2], cache.shape[-1], dtype=dtype))
        self.buffers[batch] = b
        return b

    def _slots(self, b, positions):
        page = self.decoder.block_size
        blocks = b['tables'].gather(1, positions.div(page, rounding_mode='floor'))
        return blocks * page + positions.remainder(page)

    def _stage(self, seqs):
        batch = 1 << (len(seqs) - 1).bit_length() if self.decoder.config.jetspec_cuda_graph else len(seqs)
        b = self.buffers.get(batch) or self._allocate(batch)
        d = self.decoder
        table = [[len(s)-1, s.last_token] + s.block_table +
            [d.target.pad_block_id] * (self.table_width-len(s.block_table)) for s in seqs]
        table += [[-1, 0] + [d.target.pad_block_id]*self.table_width for _ in range(batch-len(seqs))]
        b['meta'].copy_(d._host_tensor(table, torch.int64), non_blocking=True)
        b['prefix'].copy_(b['meta'][:, 0].clamp_min(0))
        b['active'].copy_(b['meta'][:, 0] >= 0)
        b['anchors'].copy_(b['meta'][:, 1])
        b['tables'].copy_(b['meta'][:, 2:])
        b['head_ids'].fill_(d.head.mask_token_id)
        b['head_ids'][:, 0].copy_(b['anchors'])
        b['head_positions'].copy_(b['prefix'][:, None] + torch.arange(self.block, device=d.device))
        b['head_slots'].copy_(self._slots(b, b['head_positions']).masked_fill(~b['active'][:, None], -1))
        b['head_lens'].copy_(b['prefix'] + self.block)
        return b

    def _forward(self, b, stage):
        d, batch = self.decoder, b['prefix'].numel()
        if stage == 'draft':
            set_context(False, slot_mapping=b['head_slots'].flatten(), context_lens=b['head_lens'],
                block_tables=b['tables'], query_len_per_seq=self.block, causal=True)
            forward = lambda: d.head(d.target.model.model.embed_tokens(b['head_ids'].flatten()),
                b['head_positions'].flatten())
        else:
            tree = PagedTreeAttention(b['tables'], b['prefix'], b['ancestors'], self.nodes)
            set_context(False, slot_mapping=b['slots'].flatten(), tree=tree)
            forward = lambda: d.target.model(b['tokens'].flatten(), b['positions'].flatten(), b['aux'])
        try:
            if not d.config.jetspec_cuda_graph:
                return forward()
            key = (stage, batch)
            if key not in self.graphs:
                # Warm the compiled operators on a side stream before capture.
                stream = torch.cuda.Stream(device=d.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    hidden = forward()
                self.graphs[key] = (graph, hidden)
                d.stats['graph_captures'] += 1
            graph, hidden = self.graphs[key]
            graph.replay()
            d.stats['graph_replays'] += 1
            return hidden
        finally:
            reset_context()

    def propose(self, b):
        d, batch = self.decoder, b['prefix'].numel()
        hidden = self._forward(b, 'draft').view(batch, self.block, -1)
        selected = hidden[:, :self.block-1] if d.config.jetspec_draft_shift else hidden[:, 1:]
        logits = d.target.model.compute_logits(selected.reshape(-1, hidden.shape[-1]), all_logits=True)
        logits = logits.view(batch, self.block-1, -1)
        build_trees(b['anchors'], logits, d.config.jetspec_tree_width,
            b['tokens'], b['parents'], b['depths'], b['ancestors'])
        b['positions'].copy_(b['prefix'][:, None] + b['depths'])
        scratch = b['prefix'][:, None] + torch.arange(self.nodes, device=d.device)
        b['slots'].copy_(self._slots(b, scratch).masked_fill(~b['active'][:, None], -1))

    def verify(self, b):
        hidden = self._forward(b, 'target')
        return self.decoder.target.model.compute_logits(hidden, all_logits=True).view(b['prefix'].numel(), self.nodes, -1)

    def accept(self, b, logits, sampling):
        if sampling.temperatures is None:
            draws = logits.argmax(-1)
        else:
            draws = sample(probs_from_logits(logits, sampling), self.decoder.target.generator)
        traverse_trees(b['tokens'], b['parents'], draws, b['path'], b['packed'], b['counts'])
        b['counts'].mul_(b['active'])

    def commit(self, b):
        d, batch = self.decoder, b['prefix'].numel()
        commit_kv(d.target.kv_cache, b['kv_temporary'], b['tables'], b['prefix'], b['path'], b['counts'])
        aux = b['aux'].view(batch, self.nodes, -1)
        accepted = aux.gather(1, b['path'][:, :, None].expand(-1, -1, d.aux_width)).reshape(-1, d.aux_width)
        positions = b['prefix'][:, None] + torch.arange(self.path_size, device=d.device)
        slots = self._slots(b, positions)
        valid = torch.arange(self.path_size, device=d.device)[None, :] < b['counts'][:, None]
        slots = slots.masked_fill(~valid, -1).to(torch.int32)
        d.head.store_context(accepted, positions.flatten(), slots.flatten())

    @torch.inference_mode()
    def run_round(self, seqs):
        d = self.decoder
        b = self._stage(seqs)
        # Padding requests use the same sampling parameters as the final real row.
        padded_seqs = seqs + [seqs[-1]] * (b['prefix'].numel() - len(seqs))
        sampling = sampling_tensors(padded_seqs, d.device)
        mark = d._mark('start', None)
        self.propose(b)
        mark = d._mark('propose_tree', mark)
        logits = self.verify(b)
        mark = d._mark('verify_tree', mark)
        self.accept(b, logits, sampling)
        mark = d._mark('accept_tree', mark)
        self.commit(b)
        d._mark('commit_tree', mark)
        # One batch transfer is required by nano's host scheduler. No per-node
        # item(), tolist(), heap, mask construction or branch decision on the CPU.
        rows = b['packed'][:len(seqs)].cpu().tolist()
        for row in rows:
            d.stats[f'accepted_{row[0]-1}'] += 1
        d.stats['proposed_nodes'] += len(seqs) * (self.nodes-1)
        d.stats['verified_positions'] += len(seqs) * self.nodes
        d.stats['rounds'] += 1
        d.stats['batched_target_forwards'] += 1
        d.stats['scheduler_batch_transfers'] += 1
        return [row[1:row[0]+1] for row in rows]

    def close(self):
        self.graphs.clear()
        self.buffers.clear()
