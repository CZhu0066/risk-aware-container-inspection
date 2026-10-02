import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import RewardModel
from utils import set_seed

RISK_RATIO_CANDIDATES = [1.00, 1.05, 1.10, 1.15, 1.20]
# RISK_RATIO_CANDIDATES = [1.05]
ALPHA = 0.5

def compute_entropy_weight(p: torch.Tensor, eps: float = 1e-6, alpha: float = ALPHA) -> torch.Tensor:
    """贝叶斯后验不确定性权重：w_unc = alpha + (1-alpha)*(1-H(p))。"""
    p_clipped = torch.clamp(p, eps, 1.0 - eps)
    h_p = -p_clipped * torch.log2(p_clipped) - (1.0 - p_clipped) * torch.log2(1.0 - p_clipped)
    return alpha + (1.0 - alpha) * (1.0 - h_p)

def compute_action_weights(chosen_action: torch.Tensor, p0: float, p1: float, risk_ratio: float) -> tuple[torch.Tensor, float, float]:

    if not (1.0 <= risk_ratio <= 1.2):
        raise ValueError("risk_ratio must in [1.0, 1.2]")
    action0_weight = 1.0 / (p0 + risk_ratio * p1)
    action1_weight = risk_ratio * action0_weight
    weights = torch.where(
        chosen_action == 1,
        torch.full_like(chosen_action, action1_weight, dtype=torch.float32),
        torch.full_like(chosen_action, action0_weight, dtype=torch.float32)
    )
    return weights, action0_weight, action1_weight

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for b_states, b_chosen, b_rejected, b_weights in dataloader:
            b_states = b_states.to(device)
            b_chosen = b_chosen.to(device)
            b_rejected = b_rejected.to(device)
            b_weights = b_weights.to(device)
            loss = model.compute_bt_loss(state=b_states, chosen_action=b_chosen, rejected_action=b_rejected, weight=b_weights)
            total_loss += loss.item() * len(b_states)
            r_c, r_r = model.forward_pair(b_states, b_chosen, b_rejected)
            correct += (r_c > r_r).sum().item()
            total += len(b_states)
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def train_reward_model(risk_ratio: float):
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = r"E:\\RLHF\\20260804\\dataset\\Dtrain_RM.csv"
    save_dir = r"E:\\RLHF\\20260916\\parameter_analysis"
    ratio_tag = f"{risk_ratio:.2f}"
    save_model_path = os.path.join(save_dir, f"reward_model_risk_{ratio_tag}.pth")
    save_scaler_path = os.path.join(save_dir, f"feature_scaler_risk_{ratio_tag}.pkl")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}，")

    df = pd.read_csv(csv_path)
    feature_cols = [
        'Max_Heatmap_Value',
        'Anomaly_Probability',
        'Mean_Anomaly_Score',
        'Anomaly_Ratio_0.25',
        'Boundary_Transition_Entropy',
        'Gradient_Strength_in_Color_Band_Direction'
    ]
    required_cols = feature_cols + ['Decision_Label', 'posterior_prob']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The dataset is missing a necessary column: {missing_cols}")

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Decision_Label'])

    scaler = StandardScaler()
    train_states_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_states_scaled = scaler.transform(val_df[feature_cols].values)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, save_scaler_path)
    print(f"Feature standardization Scaler has been saved to: {save_scaler_path}")

    train_states = torch.tensor(train_states_scaled, dtype=torch.float32)
    train_chosen = torch.tensor(train_df['Decision_Label'].values, dtype=torch.long)
    train_rejected = 1 - train_chosen
    val_states = torch.tensor(val_states_scaled, dtype=torch.float32)
    val_chosen = torch.tensor(val_df['Decision_Label'].values, dtype=torch.long)
    val_rejected = 1 - val_chosen

    p1 = float((train_chosen == 1).float().mean().item())
    p0 = 1.0 - p1
    if p0 <= 0.0 or p1 <= 0.0:
        raise ValueError("The training set for RM must contain samples with both action=0 and action=1.")

    train_uncertainty_weights = compute_entropy_weight(torch.tensor(train_df['posterior_prob'].values, dtype=torch.float32))
    val_uncertainty_weights = compute_entropy_weight(torch.tensor(val_df['posterior_prob'].values, dtype=torch.float32))

    train_action_weights, action0_weight, action1_weight = compute_action_weights(train_chosen, p0, p1, risk_ratio)
    val_action_weights, _, _ = compute_action_weights(val_chosen, p0, p1, risk_ratio)

    train_weights = train_uncertainty_weights * train_action_weights
    val_weights = val_uncertainty_weights * val_action_weights

    print("\n" + "=" * 90)
    print("RM action weight")
    print("=" * 90)
    print(f"risk_ratio      = {risk_ratio:.2f}")
    print(f"alpha           = {ALPHA:.2f}")
    print(f"Action 0 ratio  = {p0:.4f}")
    print(f"Action 1 ratio  = {p1:.4f}")
    print(f"Action 0 weight = {action0_weight:.4f}")
    print(f"Action 1 weight = {action1_weight:.4f}")
    print(f"Weighted mean   = {p0 * action0_weight + p1 * action1_weight:.4f}")
    print("=" * 90)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_states, train_chosen, train_rejected, train_weights), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_states, val_chosen, val_rejected, val_weights), batch_size=batch_size, shuffle=False)

    model = RewardModel(state_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    print(f"begin to train the reward model | risk_ratio={risk_ratio:.2f} | trainning set={len(train_df)} | validation set={len(val_df)} | device={device}")

    max_epochs = 100
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_total = 0

        for b_states, b_chosen, b_rejected, b_weights in train_loader:
            b_states = b_states.to(device)
            b_chosen = b_chosen.to(device)
            b_rejected = b_rejected.to(device)
            b_weights = b_weights.to(device)

            loss = model.compute_bt_loss(state=b_states, chosen_action=b_chosen, rejected_action=b_rejected, weight=b_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(b_states)
            train_total += len(b_states)

        avg_train_loss = train_loss / train_total
        _, train_acc = evaluate(model, train_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{max_epochs:03d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_model_path)
            print(f" Validation set Loss improve ({val_loss:.4f})，the best model is saved！")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"validation set Loss continue {patience}  Epoch not reduce，trigger Early Stopping！")
                break

    print(f"\nrisk_ratio={risk_ratio:.2f} over | Best Val Loss: {best_val_loss:.4f} | Pairwise Acc: {best_val_acc * 100:.2f}%")
    print(f"model: {save_model_path}")
    print(f"Scaler: {save_scaler_path}")

if __name__ == "__main__":
    for risk_ratio in RISK_RATIO_CANDIDATES:
        print("\n\n" + "#" * 100)
        print(f"experiment begin：risk_ratio = {risk_ratio:.2f}")
        print("#" * 100)
        train_reward_model(risk_ratio)
