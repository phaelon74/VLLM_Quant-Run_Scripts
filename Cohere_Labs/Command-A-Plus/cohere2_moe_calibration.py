"""
Cohere2 MoE calibration helpers for llm-compressor.

Command A+ (cohere2_moe) stores routed experts on disk as per-expert
`gate_proj` / `up_proj` / `down_proj` weights, but after `from_pretrained`
they are merged in memory into fused `Cohere2MoeExperts.gate_up_proj` and
`down_proj` Parameters (Mixtral-style). `QuantizationModifier` only targets
`nn.Linear`, so those Parameters are skipped unless we unfuse first.

This module replaces each `Cohere2MoeSparseMoeBlock` with a calibration variant
that exposes `mlp.experts.{i}.{gate,up,down}_proj` Linears — matching the
checkpoint layout vLLM expects (see CohereLabs/command-a-plus-05-2026-w4a4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.modeling.moe_context import MoECalibrationModule


class Cohere2ExpertMLP(nn.Module):
    """Single routed expert with separate gate/up/down projections."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        act_fn,
    ):
        super().__init__()
        self.act_fn = act_fn
        self.gate_proj = nn.Linear(
            gate_weight.shape[1], gate_weight.shape[0], bias=False
        )
        self.up_proj = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False)
        self.down_proj = nn.Linear(
            down_weight.shape[1], down_weight.shape[0], bias=False
        )
        self.gate_proj.weight = nn.Parameter(gate_weight.clone())
        self.up_proj.weight = nn.Parameter(up_weight.clone())
        self.down_proj.weight = nn.Parameter(down_weight.clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


@MoECalibrationModule.register("Cohere2MoeSparseMoeBlock")
class CalibrationCohere2MoeSparseMoeBlock(MoECalibrationModule):
    """
    Unfuse Cohere2MoeExperts Parameters into quantizable per-expert Linears.

    `is_permanent=True` keeps this layout through RTN PTQ and `save_pretrained`,
    so saved keys remain `...mlp.experts.{i}.gate_proj.weight` (etc.).
    """

    is_permanent = True

    def __init__(
        self,
        original: nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.calibrate_all_experts = calibrate_all_experts
        self.num_shared_experts = original.num_shared_experts
        self.shared_expert_combination_strategy = (
            original.shared_expert_combination_strategy
        )
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.act_fn = original.experts.act_fn

        gate_up_proj = original.experts.gate_up_proj
        down_proj = original.experts.down_proj
        num_experts = gate_up_proj.shape[0]
        intermediate_size = down_proj.shape[2]

        self.experts = nn.ModuleList()
        for expert_idx in range(num_experts):
            gate_up_fused = gate_up_proj[expert_idx]
            gate_weight = gate_up_fused[:intermediate_size, :]
            up_weight = gate_up_fused[intermediate_size:, :]
            down_weight = down_proj[expert_idx]
            self.experts.append(
                Cohere2ExpertMLP(
                    gate_weight=gate_weight,
                    up_weight=up_weight,
                    down_weight=down_weight,
                    act_fn=self.act_fn,
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        _, router_scores, selected_experts = self.gate(hidden_states_flat)

        if self.calibrate_all_experts:
            final_hidden_states = self._forward_all_experts(
                hidden_states_flat, selected_experts, router_scores
            )
        else:
            final_hidden_states = self._forward_routed(
                hidden_states_flat, selected_experts, router_scores
            )

        if self.num_shared_experts > 0:
            shared_expert_output = self.shared_experts(hidden_states_flat)
            if self.shared_expert_combination_strategy == "sum":
                final_hidden_states = final_hidden_states + shared_expert_output
            elif self.shared_expert_combination_strategy == "average":
                final_hidden_states = (
                    final_hidden_states + shared_expert_output
                ) / 2
            else:
                raise ValueError(
                    "`shared_expert_combination_strategy` must be `sum` or `average`"
                )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    def _forward_routed(
        self,
        hidden_states_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        router_scores: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=len(self.experts)
        )
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_flat[token_idx]
            expert_output = self.experts[expert_idx](current_state)
            weights = router_scores[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0,
                token_idx,
                (expert_output * weights).to(final_hidden_states.dtype),
            )
        return final_hidden_states

    def _forward_all_experts(
        self,
        hidden_states_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        router_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Run every expert on every token (for AWQ/GPTQ calibration statistics)."""
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=len(self.experts)
        ).permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            expert_output_full = expert(hidden_states_flat)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            expert_output = expert_output_full[token_idx]
            weights = router_scores[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0,
                token_idx,
                (expert_output * weights).to(final_hidden_states.dtype),
            )
        return final_hidden_states


def unfuse_cohere2_moe_experts(model: nn.Module, calibrate_all_experts: bool = False) -> int:
    """
    Permanently replace every Cohere2MoeSparseMoeBlock with the calibration variant.

    Call this before RTN `oneshot()` so routed expert weights are `nn.Linear`
    modules. Returns the number of MoE blocks replaced.
    """
    config = getattr(model, "config", None)
    replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "Cohere2MoeSparseMoeBlock":
            continue
        if "." in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0])
            child_name = name.rsplit(".", 1)[1]
        else:
            parent = model
            child_name = name
        replacement = CalibrationCohere2MoeSparseMoeBlock(
            module,
            config,
            calibrate_all_experts=calibrate_all_experts,
        )
        parent.set_submodule(child_name, replacement)
        replaced += 1
    if replaced:
        print(
            f"Unfused {replaced} Cohere2MoeSparseMoeBlock layer(s) into per-expert "
            f"gate/up/down Linears (calibrate_all_experts={calibrate_all_experts})."
        )
    return replaced
