# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py


import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MoEGate(nn.Module):
    def __init__(
        self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):

        bsz, seq_len, h = hidden_states.shape

        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # * auxiliary loss
        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:  # False
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)  # (4)
                Pi = scores_for_aux.mean(0)  # (4)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.SiLU()

        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(
                slice, dim=-1
            )
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        embed_dim,
        mlp_ratio=4,
        num_experts=16,
        num_experts_per_tok=2,
        pretraining_tp=2,
        n_shared_experts=2,
        num_routers=1,
        use_pre_norm=False,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                MoeMLP(
                    hidden_size=embed_dim,
                    intermediate_size=mlp_ratio * embed_dim,
                    pretraining_tp=pretraining_tp,
                )
                for i in range(num_experts)
            ]
        )

        self.gates = nn.ModuleList(
            [
                MoEGate(
                    embed_dim=embed_dim,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                )
                for _ in range(num_routers)
            ]
        )

        self.use_pre_norm = use_pre_norm
        if self.use_pre_norm:
            # Each DINO feature level
            # has its own learnable LayerNorm. Routers and routed experts consume
            # normalized tokens, while the residual path preserves the raw tokens.
            self.moe_norms = nn.ModuleList(
                [nn.LayerNorm(embed_dim) for _ in range(num_routers)]
            )

        self.n_shared_experts = n_shared_experts
        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp,
            )

    def forward(self, hidden_states, router_idx=0, **kwargs):
        identity = hidden_states  # unnormalized residual
        orig_shape = hidden_states.shape

        if self.use_pre_norm:
            hidden_states = self.moe_norms[router_idx](hidden_states)
        topk_idx, topk_weight, aux_loss = self.gates[router_idx](hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # expert input
        flat_topk_idx = topk_idx.view(-1)  # (2400)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(
                    hidden_states[flat_topk_idx == i]
                ).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)

        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        else:
            y = y + identity
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache
