"""
NemotronH (Nemotron 3 Super) MoE Calibration Module for llmcompressor

NemotronHMoE uses nn.ModuleList of NemotronHMLP experts (each has nn.Linear up_proj/down_proj).
Routing params (n_routed_experts, n_group, etc.) live on the gate (NemotronHTopkRouter), not the MoE.
This calibration module:
1. Wraps original experts into NemotronExpertMLP for consistent naming and all-expert activation
2. Runs ALL tokens through ALL experts during calibration (expert activation for AWQ)
3. Produces weight names: experts.{i}.up_proj, experts.{i}.down_proj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from llmcompressor.modeling.moe_context import MoECalibrationModule


class NemotronExpertMLP(nn.Module):
    """
    Single expert MLP with up_proj and down_proj (no gate - standard MLP).
    NemotronH uses relu2 activation.
    """

    def __init__(self, up_weight: torch.Tensor, down_weight: torch.Tensor, act_fn):
        super().__init__()
        self.act_fn = act_fn
        # up_proj: [intermediate_dim, input_dim] -> Linear(input_dim, intermediate_dim)
        self.up_proj = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False)
        self.up_proj.weight = nn.Parameter(up_weight.clone())
        # down_proj: [input_dim, intermediate_dim] -> Linear(intermediate_dim, input_dim)
        self.down_proj = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        self.down_proj.weight = nn.Parameter(down_weight.clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


@MoECalibrationModule.register("NemotronHMoE")
class CalibrationNemotronHMoE(MoECalibrationModule):
    """
    Calibration version of NemotronHMoE that wraps experts for AWQ calibration.
    All experts receive all tokens during forward (expert activation for calibration).
    """

    is_permanent = True
    _forward_call_count = 0
    _last_log_count = 0

    def __init__(
        self,
        original: nn.Module,
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.fc1_latent_proj = original.fc1_latent_proj
        self.fc2_latent_proj = original.fc2_latent_proj
        self.calibrate_all_experts = calibrate_all_experts

        # Routing params live on the gate (NemotronHTopkRouter), not the MoE
        self.n_routed_experts = original.gate.n_routed_experts
        self.n_group = original.gate.n_group
        self.topk_group = original.gate.topk_group
        self.norm_topk_prob = original.gate.norm_topk_prob
        self.routed_scaling_factor = original.gate.routed_scaling_factor
        self.top_k = original.gate.top_k

        # NemotronH experts are nn.ModuleList of NemotronHMLP (each has up_proj, down_proj as nn.Linear)
        orig_experts = original.experts
        num_experts = len(orig_experts)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            expert = orig_experts[i]
            up_w = expert.up_proj.weight
            down_w = expert.down_proj.weight
            expert_mlp = NemotronExpertMLP(
                up_weight=up_w,
                down_weight=down_w,
                act_fn=expert.act_fn,
            )
            self.experts.append(expert_mlp)

        print(f"[CalibrationNemotronHMoE] Created {len(self.experts)} experts (up_proj, down_proj)")
        print(f"  Naming pattern: experts.{{i}}.up_proj, experts.{{i}}.down_proj")

    def route_tokens_to_experts(self, router_logits):
        """Replicates NemotronHMoE routing logic."""
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        CalibrationNemotronHMoE._forward_call_count += 1
        if CalibrationNemotronHMoE._forward_call_count - CalibrationNemotronHMoE._last_log_count >= 1000:
            print(f"[CalibrationNemotronHMoE] forward() called {CalibrationNemotronHMoE._forward_call_count} times")
            CalibrationNemotronHMoE._last_log_count = CalibrationNemotronHMoE._forward_call_count

        residuals = hidden_states
        orig_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])

        # Gate.forward returns (topk_indices, topk_weights); we need router_logits for route_tokens_to_experts
        router_logits = F.linear(
            hidden_states_flat.type(torch.float32),
            self.gate.weight.type(torch.float32),
        )
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        hidden_states = self.fc1_latent_proj(hidden_states)
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_experts = len(self.experts)

        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=num_experts
        ).permute(2, 0, 1)

        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            expert_output_full = expert(hidden_states_flat)

            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_output = expert_output_full[token_indices]
                expert_weights = topk_weights[token_indices, weight_indices].to(expert_output.dtype)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))

        hidden_states = final_hidden_states.view(*orig_shape)
        hidden_states = self.fc2_latent_proj(hidden_states)
        hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states
