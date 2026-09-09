"""
GLM-4.7-Flash (glm4_moe_lite) MoE Calibration Module for llmcompressor - V2

CRITICAL FIX: This version produces vLLM-compatible weight names!

The key insight from working quantized models (cyankiwi/GLM-4.7-Flash-AWQ-4bit):
- They use: model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight_packed
- NOT: model.layers.{L}.mlp.expert_gate_up.{E}.weight_packed (our old format)

This version:
1. Creates separate gate_proj, up_proj, down_proj Linear modules per expert (NOT fused)
2. Uses the naming pattern `experts.{i}.{gate_proj|up_proj|down_proj}`
3. Produces weights that vLLM's glm4_moe_lite.py can load directly
"""

import torch
import torch.nn as nn
from llmcompressor.modeling.moe_context import MoECalibrationModule


class ExpertMLP(nn.Module):
    """
    A single expert MLP with separate gate_proj, up_proj, and down_proj.
    
    This matches the naming convention that vLLM expects:
        experts.{i}.gate_proj
        experts.{i}.up_proj
        experts.{i}.down_proj
    """
    
    def __init__(self, gate_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor, act_fn):
        """
        Initialize with separate gate, up, and down projection weights.
        
        Args:
            gate_weight: [intermediate_size, hidden_size] - gate projection
            up_weight: [intermediate_size, hidden_size] - up projection
            down_weight: [hidden_size, intermediate_size] - down projection
            act_fn: Activation function (e.g., SiLU)
        """
        super().__init__()
        self.act_fn = act_fn
        
        # Create separate Linear modules - these will be quantized individually
        # gate_proj: hidden -> intermediate
        self.gate_proj = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        self.gate_proj.weight = nn.Parameter(gate_weight.clone())
        
        # up_proj: hidden -> intermediate
        self.up_proj = nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False)
        self.up_proj.weight = nn.Parameter(up_weight.clone())
        
        # down_proj: intermediate -> hidden
        self.down_proj = nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False)
        self.down_proj.weight = nn.Parameter(down_weight.clone())
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU-style forward pass with separate gate and up projections.
        """
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        intermediate = gate * up
        return self.down_proj(intermediate)


@MoECalibrationModule.register("Glm4MoeLiteMoE")
class CalibrationGlm4MoeLiteMoE(MoECalibrationModule):
    """
    Calibration version of Glm4MoeLiteMoE that produces vLLM-compatible weights.
    
    Key changes from v1:
    1. Uses `self.experts` ModuleList containing ExpertMLP modules
    2. Each ExpertMLP has separate gate_proj, up_proj, down_proj
    3. This produces weight names like: experts.0.gate_proj.weight_packed
    4. vLLM can load these directly!
    
    Architecture details:
    - Glm4MoeLiteNaiveMoe stores expert weights as fused 3D tensors:
      - gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
      - down_proj: [num_experts, hidden_size, intermediate_size]
    - We SPLIT gate_up into separate gate and up for each expert
    """
    
    is_permanent = True  # Keep wrappers during quantization
    _forward_call_count = 0
    _last_log_count = 0
    
    def __init__(
        self,
        original: nn.Module,  # Glm4MoeLiteMoE
        config,
        calibrate_all_experts: bool = True,
    ):
        super().__init__()
        self.config = config
        self.gate = original.gate
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts
        
        # Copy routing parameters
        self.n_routed_experts = original.n_routed_experts
        self.n_group = original.n_group
        self.topk_group = original.topk_group
        self.norm_topk_prob = original.norm_topk_prob
        self.routed_scaling_factor = original.routed_scaling_factor
        self.top_k = original.top_k
        self.act_fn = original.experts.act_fn
        
        # Store num_experts for forward pass
        self.num_experts = original.experts.num_experts
        
        # Get dimensions from the fused tensors
        # gate_up_proj: [num_experts, 2*intermediate, hidden]
        # down_proj: [num_experts, hidden, intermediate]
        gate_up_proj = original.experts.gate_up_proj
        down_proj = original.experts.down_proj
        
        intermediate_size = down_proj.shape[2]  # intermediate dimension
        
        # Create ModuleList of ExpertMLP modules with SEPARATE gate, up, down
        # This produces the naming pattern: experts.{i}.{gate_proj|up_proj|down_proj}
        self.experts = nn.ModuleList()
        
        for expert_idx in range(self.num_experts):
            # Get the fused gate_up for this expert and SPLIT it
            gate_up_fused = gate_up_proj[expert_idx]  # [2*intermediate, hidden]
            
            # Split into separate gate and up weights
            gate_weight = gate_up_fused[:intermediate_size, :]   # [intermediate, hidden]
            up_weight = gate_up_fused[intermediate_size:, :]     # [intermediate, hidden]
            
            # Get down projection for this expert
            down_weight = down_proj[expert_idx]  # [hidden, intermediate]
            
            # Create ExpertMLP with separate projections
            expert_mlp = ExpertMLP(
                gate_weight=gate_weight,
                up_weight=up_weight,
                down_weight=down_weight,
                act_fn=self.act_fn
            )
            self.experts.append(expert_mlp)
        
        print(f"[CalibrationGlm4MoeLiteMoE v2] Created {self.num_experts} experts with separate gate/up/down projections")
        print(f"  Naming pattern: experts.{{i}}.gate_proj, experts.{{i}}.up_proj, experts.{{i}}.down_proj")
    
    def route_tokens_to_experts(self, router_logits):
        """Replicates the routing logic from Glm4MoeLiteMoE."""
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
        """
        Forward pass using ExpertMLP modules.
        
        All tokens are sent to all experts during calibration to ensure
        proper calibration statistics for quantization.
        """
        CalibrationGlm4MoeLiteMoE._forward_call_count += 1
        if CalibrationGlm4MoeLiteMoE._forward_call_count - CalibrationGlm4MoeLiteMoE._last_log_count >= 1000:
            print(f"[CalibrationGlm4MoeLiteMoE v2] forward() called {CalibrationGlm4MoeLiteMoE._forward_call_count} times")
            CalibrationGlm4MoeLiteMoE._last_log_count = CalibrationGlm4MoeLiteMoE._forward_call_count
        
        residuals = hidden_states
        orig_shape = hidden_states.shape
        
        # Get routing decisions
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_experts = len(self.experts)
        
        # Initialize output
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        # Create expert mask for proper weighting
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=num_experts
        ).permute(2, 0, 1)  # [num_experts, num_tokens, top_k]
        
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            
            # Run ALL tokens through this expert for calibration
            # The ExpertMLP.forward() calls gate_proj, up_proj, down_proj
            # which allows GPTQ/AWQ hooks to attach and collect statistics
            expert_output_full = expert(hidden_states_flat)
            
            # Apply routing weights for final output
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            
            if token_indices.numel() > 0:
                expert_output = expert_output_full[token_indices]
                expert_weights = topk_weights[token_indices, weight_indices].to(expert_output.dtype)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output.to(final_hidden_states.dtype))
        
        hidden_states = final_hidden_states.view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        
        return hidden_states
