import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from CF_risk_regulazition import CognitiveCounterfactualModule
from model import RewardModel, PPOActorCritic, compute_gae
from utils import set_seed

def train_ppo():
    seed = 43
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths
    csv_path = r"E:\\RLHF\\20260804\\dataset\\Dtrain_PPO.csv"
    rm_model_path = r"E:\\RLHF\\20260916\\parameter_analysis\\reward_model_risk_1.20.pth"
    scaler_path = r"E:\\RLHF\\20260916\\parameter_analysis\\feature_scaler_risk_1.20.pkl"
    save_ppo_path = r"E:\\RLHF\\20260916\\parameter_analysis\\ppo_lamda_0.3.pth"
    scm_params_path = r"E:\\RLHF\\20260916\\parameter_analysis\\scm_params.pkl"

    required_files = [csv_path, rm_model_path, scaler_path, scm_params_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load data
    df = pd.read_csv(csv_path)
    scaler = joblib.load(scaler_path)

    feature_cols = [
        'Max_Heatmap_Value',
        'Anomaly_Probability',
        'Mean_Anomaly_Score',
        'Anomaly_Ratio_0.25',
        'Boundary_Transition_Entropy',
        'Gradient_Strength_in_Color_Band_Direction'
    ]

    states_scaled = scaler.transform(df[feature_cols].values)

    # 2. Load the Reward Model
    reward_model = RewardModel(state_dim=len(feature_cols)).to(device)
    reward_model.load_state_dict(torch.load(rm_model_path, map_location=device))
    reward_model.eval()

    # Freeze the RM so that it does not participate in PPO parameter updates
    for param in reward_model.parameters():
        param.requires_grad_(False)

    # 3. Initialize PPO
    ppo_agent = PPOActorCritic(state_dim=len(feature_cols), action_dim=2).to(device)

    # 4. Initialize the counterfactual module
    cf_module = CognitiveCounterfactualModule(
        scm_params_path=scm_params_path,
        action0_risk_weight=1.0,
        action1_risk_weight=0.4,
        margin_threshold_action0=0.70,
        margin_threshold_action1=0.30,
        lr=0.02,
        max_steps=6,
        early_stop_patience=2,
        boundary_refine_steps=10
    )

    lambda_cf = 0.3

    # 5. PPO optimizers
    actor_optimizer = torch.optim.Adam(ppo_agent.actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(ppo_agent.critic.parameters(), lr=1e-3)

    # 6. PPO hyperparameters
    epochs = 40
    ppo_epochs = 4
    clip_eps = 0.2
    batch_size = 64
    best_reward = -float("inf")

    print(f"🚀 Starting PPO agent training (Samples/Epoch: {len(states_scaled)} | Device: {device})...")

    for epoch in range(1, epochs + 1):
        b_states = []
        b_sampled_actions = []
        b_log_probs = []
        b_values = []
        b_rewards = []
        b_dones = []

        b_rm_rewards = []
        b_rm_preference_gaps = []
        b_cf_penalties = []
        b_policy_cf_triggered = []
        b_cf_triggered = []
        b_cf_valids = []
        b_cf_distances = []

        # =========================================================
        # Stage 1: randomly shuffle the entire PPO training set and traverse it once per epoch
        # =========================================================
        indices = np.random.permutation(len(states_scaled))

        for idx in indices:
            state = states_scaled[idx]
            state_t = torch.tensor(state, dtype=torch.float32, device=device)

            # -----------------------------------------------------
            # 1. PPO rollout: continue using stochastically sampled actions
            # -----------------------------------------------------
            with torch.no_grad():
                sampled_action, log_prob, value = ppo_agent.select_action(state_t, device)
                policy_probs = ppo_agent.get_action_probs(state_t.unsqueeze(0)).squeeze(0)

            # The highest-probability action is used only for CF risk analysis
            policy_action = torch.argmax(policy_probs).item()
            policy_confidence = policy_probs[policy_action].item()
            policy_margin = torch.abs(policy_probs[1] - policy_probs[0]).item()

            # =====================================================
            # 2. Reward Model reward and preference conflict
            # =====================================================
            with torch.no_grad():
                sampled_action_tensor = torch.tensor([sampled_action], dtype=torch.float32, device=device)

                # RM reward for the action actually sampled by PPO
                rm_reward = reward_model.compute_action_reward(
                    state=state_t.unsqueeze(0),
                    action=sampled_action_tensor,
                    temperature=1.0
                ).item()

                # Compute the RM reward for both actions separately
                action0_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)
                action1_tensor = torch.tensor([1.0], dtype=torch.float32, device=device)

                rm_reward_action0 = reward_model.compute_action_reward(
                    state=state_t.unsqueeze(0),
                    action=action0_tensor,
                    temperature=1.0
                ).item()

                rm_reward_action1 = reward_model.compute_action_reward(
                    state=state_t.unsqueeze(0),
                    action=action1_tensor,
                    temperature=1.0
                ).item()

                # > 0: RM prefers action=1
                # < 0: RM prefers action=0
                rm_preference_gap = rm_reward_action1 - rm_reward_action0

            # =====================================================
            # 3. CF risk evaluation
            # Current C_module trigger rule:
            # trigger when policy_action=0 and the RM prefers action=1
            # =====================================================
            policy_cf_triggered = cf_module.should_trigger_cf(
                policy_action=policy_action,
                policy_margin=policy_margin,
                rm_preference_gap=rm_preference_gap
            )

            cf_triggered = False
            is_valid = False
            best_d_cf = float("nan")
            vulnerability = float("nan")
            cf_penalty = 0.0

            # The CF penalty must be consistent with the action actually sampled by PPO
            if sampled_action == policy_action and policy_cf_triggered:
                cf_triggered = True

                best_d_cf, is_valid, best_s_cf = cf_module.optimize_counterfactual(
                    ppo_actor=ppo_agent,
                    single_state=state_t,
                    policy_action=policy_action
                )

                cf_penalty, vulnerability = cf_module.compute_vulnerability_penalty(
                    policy_action=policy_action,
                    is_valid=is_valid,
                    d_cf=best_d_cf
                )

            # =====================================================
            # 4. Final reward
            # =====================================================
            final_reward = rm_reward + lambda_cf * cf_penalty

            # Contextual Bandit:
            # Each sample is an independent single-step episode
            done = True

            # =====================================================
            # 5. Store the current transition
            # =====================================================
            b_states.append(state)
            b_sampled_actions.append(sampled_action)
            b_log_probs.append(log_prob.item())
            b_values.append(value.item())
            b_rewards.append(final_reward)
            b_dones.append(float(done))

            b_rm_rewards.append(rm_reward)
            b_rm_preference_gaps.append(rm_preference_gap)
            b_cf_penalties.append(cf_penalty)
            b_policy_cf_triggered.append(float(policy_cf_triggered))
            b_cf_triggered.append(float(cf_triggered))
            b_cf_valids.append(float(is_valid))
            b_cf_distances.append(best_d_cf)

        # =========================================================
        # Average reward of the current rollout
        # =========================================================
        avg_reward = np.mean(b_rewards)

        # Save the best-performing policy so far
        if avg_reward > best_reward:
            best_reward = avg_reward
            os.makedirs(os.path.dirname(save_ppo_path), exist_ok=True)
            torch.save(ppo_agent.state_dict(), save_ppo_path)
            print(f"Saved the best PPO model (Reward={best_reward:.4f})")

        # =========================================================
        # Stage 2: GAE
        # =========================================================
        t_states = torch.tensor(np.array(b_states), dtype=torch.float32, device=device)
        t_sampled_actions = torch.tensor(b_sampled_actions, dtype=torch.long, device=device)
        t_log_probs = torch.tensor(b_log_probs, dtype=torch.float32, device=device)
        t_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        t_rewards = torch.tensor(b_rewards, dtype=torch.float32)
        t_dones = torch.tensor(b_dones, dtype=torch.float32)

        # Since done=True at every step, GAE reduces to a single-step advantage in the contextual bandit setting
        next_value = 0.0
        advantages, returns = compute_gae(
            rewards=t_rewards,
            values=t_values,
            next_value=next_value,
            dones=t_dones
        )

        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(device)
        returns = returns.to(device)

        # =========================================================
        # Stage 3: PPO parameter updates
        # =========================================================
        dataset = TensorDataset(
            t_states,
            t_sampled_actions,
            t_log_probs,
            returns,
            advantages
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        for _ in range(ppo_epochs):
            for mb_states, mb_sampled_actions, mb_old_log_probs, mb_returns, mb_advantages in dataloader:
                new_log_probs, new_values, entropy = ppo_agent.evaluate_actions(
                    mb_states,
                    mb_sampled_actions
                )

                ratios = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(
                    ratios,
                    1.0 - clip_eps,
                    1.0 + clip_eps
                ) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                critic_loss = nn.MSELoss()(new_values, mb_returns)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

        # =========================================================
        # Training statistics
        # =========================================================
        avg_rm_reward = np.mean(b_rm_rewards)
        avg_rm_preference_gap = np.mean(b_rm_preference_gaps)
        avg_cf_penalty = np.mean(b_cf_penalties)
        policy_cf_trigger_rate = np.mean(b_policy_cf_triggered)
        cf_applied_rate = np.mean(b_cf_triggered)

        n_cf_applied = np.sum(b_cf_triggered)

        if n_cf_applied > 0:
            cf_valid_rate = np.sum(b_cf_valids) / n_cf_applied
        else:
            cf_valid_rate = float("nan")

        valid_distances = [
            d for d in b_cf_distances
            if np.isfinite(d)
        ]

        avg_cf_distance = (
            np.mean(valid_distances)
            if len(valid_distances) > 0
            else float("nan")
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Final Reward: {avg_reward:.3f} | "
            f"RM Reward: {avg_rm_reward:.3f} | "
            f"RM Gap: {avg_rm_preference_gap:.3f} | "
            f"CF Penalty: {avg_cf_penalty:.3f} | "
            f"Policy CF Trigger Rate: {policy_cf_trigger_rate * 100:.2f}% | "
            f"CF Applied Rate: {cf_applied_rate * 100:.2f}% | "
            f"CF Valid Rate: {cf_valid_rate * 100:.2f}% | "
            f"Mean d_cf: {avg_cf_distance:.3f}"
        )

    print(f"PPO model training completed. Best model saved to: {save_ppo_path}")

if __name__ == "__main__":
    train_ppo()