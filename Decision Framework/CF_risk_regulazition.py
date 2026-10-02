import joblib
import os
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class CognitiveCounterfactualModule:

    def __init__(
            self,
            scm_params_path: str,
            action0_risk_weight: float = 1.0,
            action1_risk_weight: float = 0.4,
            margin_threshold_action0: float = 0.70,
            margin_threshold_action1: float = 0.30,
            lr: float = 0.02,
            max_steps: int = 6,
            early_stop_patience: int = 2,
            boundary_refine_steps: int = 10,
    ):

        if not os.path.exists(scm_params_path):
            raise FileNotFoundError(
                f"SCM parameter file not found: {scm_params_path}"
            )
        scm_params = joblib.load(scm_params_path)

        self.tolerance = torch.tensor(scm_params["tolerance"],dtype=torch.float32)

        self.parents = scm_params["parents"]

        self.coefficients = scm_params["coefficients"]

        self.topological_order = scm_params["topological_order"]

        self.state_dim = len(scm_params["feature_names"])

        if not (0.0 <= action1_risk_weight<= action0_risk_weight<= 1.0):
            raise ValueError("Risk weights must satisfy: 0 <= action1_risk_weight <= action0_risk_weight <= 1")
        if not (0.0 <= margin_threshold_action1<= margin_threshold_action0<= 1.0):
            raise ValueError("CF trigger thresholds must satisfy: 0 <= margin_threshold_action1 <= margin_threshold_action0 <= 1")
        self.margin_threshold_action0 = margin_threshold_action0
        self.margin_threshold_action1 = margin_threshold_action1
        self.action0_risk_weight = action0_risk_weight
        self.action1_risk_weight = action1_risk_weight

        self.lr = lr
        self.max_steps = max_steps
        self.early_stop_patience = early_stop_patience
        self.boundary_refine_steps = boundary_refine_steps
        if self.boundary_refine_steps < 0:
            raise ValueError("boundary_refine_steps must be >= 0")

    def evaluate_policy_decision(self,ppo_actor: nn.Module,single_state: torch.Tensor):

        was_training = ppo_actor.training
        ppo_actor.eval()

        with torch.no_grad():
            probs = ppo_actor.get_action_probs(single_state.unsqueeze(0)).squeeze(0)
            policy_action = torch.argmax(probs).item()
            policy_confidence = probs[policy_action].item()
            policy_margin = torch.abs(probs[1] - probs[0]).item()

        if was_training:
            ppo_actor.train()
        return (policy_action, policy_confidence, policy_margin, probs)

    def _apply_causal_scm(
            self,
            s_orig: torch.Tensor,
            delta_s: torch.Tensor
    ) -> torch.Tensor:

        device = delta_s.device
        tol = self.tolerance.to(device)

        delta_clamped = torch.clamp(delta_s,-tol,tol)

        effective_delta = [None] * self.state_dim

        for node in self.topological_order:

            delta_node = delta_clamped[..., node]

            parent_list = self.parents[node]

            for parent in parent_list:
                beta = self.coefficients[node][parent]
                delta_node = (
                        delta_node
                        + beta * effective_delta[parent]
                )
            effective_delta[node] = delta_node

        delta_final = torch.stack(effective_delta,dim=-1)

        delta_final = torch.clamp(delta_final,-tol,tol)

        s_cf = s_orig + delta_final
        return s_cf

    def _refine_counterfactual_boundary(
            self,
            ppo_actor: nn.Module,
            single_state: torch.Tensor,
            valid_delta_s: torch.Tensor,
            policy_action: int
    ) -> Tuple[float, torch.Tensor]:
        device = single_state.device
        tol = self.tolerance.to(device)
        tol_safe = torch.clamp(tol,min=1e-6)
        low = 0.0
        high = 1.0
        with torch.no_grad():

            refined_s_cf = self._apply_causal_scm(
                s_orig=single_state,
                delta_s=valid_delta_s
            ).detach().clone()

            for _ in range(self.boundary_refine_steps):
                mid = 0.5 * (low + high)

                candidate_delta = valid_delta_s * mid

                candidate_s = self._apply_causal_scm(
                    s_orig=single_state,
                    delta_s=candidate_delta
                )

                probs_mid = ppo_actor.get_action_probs(candidate_s.unsqueeze(0)).squeeze(0)
                candidate_action = torch.argmax(probs_mid).item()

                if candidate_action != policy_action:

                    high = mid
                    refined_s_cf = (candidate_s.detach().clone())
                else:

                    low = mid

            normalized_change = ((refined_s_cf - single_state)/ tol_safe)
            refined_d_cf = torch.mean(torch.abs(normalized_change)).item()
        return float(refined_d_cf), refined_s_cf

    def optimize_counterfactual(
            self,
            ppo_actor: nn.Module,
            single_state: torch.Tensor,
            policy_action: int
    ) -> Tuple[float, bool, Optional[torch.Tensor]]:

        device = single_state.device
        tol = self.tolerance.to(device)

        tol_safe = torch.clamp(tol, min=1e-6)

        target_action = 1 - policy_action

        was_training = ppo_actor.training
        ppo_actor.eval()
        actor_requires_grad = [
            p.requires_grad
            for p in ppo_actor.actor.parameters()
        ]
        for p in ppo_actor.actor.parameters():
            p.requires_grad_(False)

        delta_s = torch.zeros_like(
            single_state,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        optimizer = torch.optim.Adam([delta_s],lr=self.lr)

        best_d_cf = float("inf")
        best_s_cf = None
        best_delta_s = None
        is_valid = False

        no_improve_count = 0

        for step in range(self.max_steps + 1):

            s_cf = self._apply_causal_scm(
                s_orig=single_state,
                delta_s=delta_s
            )

            probs_cf = ppo_actor.get_action_probs(
                s_cf.unsqueeze(0)
            ).squeeze(0)
            cf_action = torch.argmax(probs_cf).item()

            current_valid = (cf_action != policy_action)
            if current_valid:
                normalized_change = ((s_cf - single_state) / tol_safe)
                current_d_cf = torch.mean(torch.abs(normalized_change)).item()
                if current_d_cf < best_d_cf - 1e-6:
                    best_d_cf = current_d_cf
                    best_s_cf = (s_cf.detach().clone())
                    best_delta_s = (delta_s.detach().clone())
                    is_valid = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:

                if is_valid:
                    no_improve_count += 1
            if (is_valid and no_improve_count >= self.early_stop_patience):
                break

            if step == self.max_steps:
                break

            validity_loss = torch.clamp(probs_cf[policy_action] - probs_cf[target_action],min=0.0)

            normalized_change = ((s_cf - single_state) / tol_safe)
            distance_loss = torch.mean(torch.abs(normalized_change))

            total_loss = (10.0 * validity_loss+ 1.0 * distance_loss)

            optimizer.zero_grad()
            grad_delta = torch.autograd.grad(
                total_loss,
                delta_s,
                retain_graph=False,
                create_graph=False
            )[0]
            delta_s.grad = grad_delta
            optimizer.step()

        if (is_valid and best_delta_s is not None and self.boundary_refine_steps > 0):
            best_d_cf, best_s_cf = (
                self._refine_counterfactual_boundary(
                    ppo_actor=ppo_actor,
                    single_state=single_state,
                    valid_delta_s=best_delta_s,
                    policy_action=policy_action
                )
            )

        for p, requires_grad in zip(
                ppo_actor.actor.parameters(),
                actor_requires_grad
        ):
            p.requires_grad_(requires_grad)
        if was_training:
            ppo_actor.train()

        if not is_valid:
            return float("nan"), False, None

        return float(best_d_cf), True, best_s_cf

    def compute_vulnerability_penalty(
            self,
            policy_action: int,
            is_valid: bool,
            d_cf: float
    ) -> Tuple[float, float]:

        if policy_action not in (0, 1):
            raise ValueError(f"policy_action must be 0 or 1; current value: {policy_action}")

        if not is_valid:
            return 0.0, float("nan")

        if not math.isfinite(d_cf):
            raise ValueError("When is_valid=True, d_cf must be a finite value.")

        d_cf = max(0.0,min(float(d_cf), 1.0))

        vulnerability = 1.0 - d_cf

        if policy_action == 0:
            risk_weight = self.action0_risk_weight
        else:
            risk_weight = self.action1_risk_weight

        cf_penalty = -risk_weight * vulnerability
        return float(cf_penalty), float(vulnerability)

    def should_trigger_cf(
            self,
            policy_action: int,
            policy_margin: float,
            rm_preference_gap: Optional[float] = None
    ) -> bool:

        if policy_action not in (0, 1):
            raise ValueError(f"policy_action must be 0 or 1; current value: {policy_action}")

        if policy_action == 0:
            rm_conflict_trigger = (rm_preference_gap is not None and rm_preference_gap > 0.0)
            return rm_conflict_trigger

        return False