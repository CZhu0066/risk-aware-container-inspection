import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
from CF_risk_regulazition import CognitiveCounterfactualModule
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,matthews_corrcoef
from model import RewardModel, PPOActorCritic



def safe_mean(values) -> float:

    values = np.asarray(values, dtype=np.float64)

    valid_values = values[np.isfinite(values)]

    if len(valid_values) == 0:
        return float("nan")
    return float(np.mean(valid_values))


def compute_evaluation_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cf_triggered: np.ndarray,
        cf_valids: np.ndarray,
        d_cfs: np.ndarray,
        vulnerabilities: np.ndarray,
        cf_penalties: np.ndarray,
        cf_times_ms: np.ndarray
) -> dict:

    tn, fp, fn, tp = confusion_matrix(y_true,y_pred,labels=[0, 1]).ravel()
    total = len(y_true)
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score( y_true,y_pred,zero_division=0)
    recall = recall_score(y_true,y_pred,zero_division=0)
    f1 = f1_score(y_true,y_pred,zero_division=0)
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true,y_pred)

    # False Positive Rate
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = float("nan")

    # False Negative Rate
    if (fn + tp) > 0:
        fnr = fn / (fn + tp)
    else:
        fnr = float("nan")

    # Balanced Accuracy
    if np.isfinite(fpr):
        specificity = 1.0 - fpr
        balanced_accuracy = (recall + specificity) / 2.0
    else:
        balanced_accuracy = float("nan")


    trigger_mask = cf_triggered.astype(bool)
    n_triggered = int(np.sum(trigger_mask))
    if total > 0:
        cf_trigger_rate = n_triggered / total
    else:
        cf_trigger_rate = float("nan")

    valid_mask = (trigger_mask& cf_valids.astype(bool))
    n_valid = int(np.sum(valid_mask))

    if n_triggered > 0:
        cf_valid_rate = n_valid / n_triggered
    else:
        cf_valid_rate = float("nan")


    valid_d_cfs = d_cfs[valid_mask]
    mean_d_cf = safe_mean(valid_d_cfs)

    valid_vulnerabilities = vulnerabilities[valid_mask]
    mean_vulnerability = safe_mean(valid_vulnerabilities)

    triggered_penalties = cf_penalties[trigger_mask]
    mean_cf_penalty = safe_mean(triggered_penalties)

    triggered_cf_times = cf_times_ms[trigger_mask]
    mean_cf_time_ms = safe_mean(triggered_cf_times)

    return {
        "Sample Count": total,
        "Accuracy(%)": accuracy * 100.0,
        "Balanced Accuracy(%)": (balanced_accuracy * 100.0
            if np.isfinite(balanced_accuracy)
            else float("nan")
        ),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "FPR(%)": fpr * 100.0,
        "FNR(%)": fnr * 100.0,

        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "CF Trigger Count": n_triggered,
        "CF Trigger Rate(%)": (cf_trigger_rate * 100.0),
        "CF Valid Count": n_valid,
        "CF Valid Rate(%)": (
            cf_valid_rate * 100.0
            if np.isfinite(cf_valid_rate)
            else float("nan")
        ),
        "Mean d_cf": mean_d_cf,
        "Mean Vulnerability": mean_vulnerability,
        "Mean CF Penalty": mean_cf_penalty,
        "Mean CF Time(ms)": mean_cf_time_ms
    }

def compute_cf_group_metrics(
        group_mask: np.ndarray,
        cf_triggered: np.ndarray,
        cf_valids: np.ndarray,
        d_cfs: np.ndarray
) -> dict:

    n_group = int(np.sum(group_mask))
    if n_group == 0:
        return {
            "Count": 0,
            "CF Trigger Count": 0,
            "CF Trigger Rate(%)": float("nan"),
            "CF Valid Count": 0,
            "CF Valid Rate(%)": float("nan"),
            "Mean d_cf": float("nan")
        }

    trigger_mask = group_mask & cf_triggered
    n_triggered = int(np.sum(trigger_mask))
    cf_trigger_rate = n_triggered / n_group

    valid_mask = trigger_mask & cf_valids
    n_valid = int(np.sum(valid_mask))
    if n_triggered > 0:
        cf_valid_rate = n_valid / n_triggered
    else:
        cf_valid_rate = float("nan")

    mean_d_cf = safe_mean(d_cfs[valid_mask])
    return {
        "Count": n_group,
        "CF Trigger Count": n_triggered,
        "CF Trigger Rate(%)": cf_trigger_rate * 100.0,
        "CF Valid Count": n_valid,
        "CF Valid Rate(%)": (
            cf_valid_rate * 100.0
            if np.isfinite(cf_valid_rate)
            else float("nan")
        ),
        "Mean d_cf": mean_d_cf
    }

def evaluate_ppo_model():
    device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

    test_csv_path = (r"E:\\RLHF\\20260804\\dataset\\Dtest_PPO.csv")
    scaler_path = (r"E:\RLHF\20260916\parameter_analysis\feature_scaler_risk_1.20.pkl")
    ppo_model_path = (r"E:\\RLHF\\20260916\\parameter_analysis\\ppo_lamda_0.3.pth")
    rm_model_path = (r"E:\RLHF\20260916\parameter_analysis\reward_model_risk_1.20.pth")
    scm_params_path = (r"E:\\RLHF\\20260916\\parameter_analysis\\scm_params.pkl")

    required_files = [test_csv_path,scaler_path,ppo_model_path,rm_model_path,scm_params_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" Required evaluation file not found: {file_path}")

    df_test = pd.read_csv(test_csv_path)
    scaler = joblib.load(scaler_path)
    feature_cols = [
        "Max_Heatmap_Value",
        "Anomaly_Probability",
        "Mean_Anomaly_Score",
        "Anomaly_Ratio_0.25",
        "Boundary_Transition_Entropy",
        "Gradient_Strength_in_Color_Band_Direction"
    ]

    required_columns = (feature_cols+ ["Decision_Label","Class_Index"])
    for col in required_columns:
        if col not in df_test.columns:
            raise ValueError(f"Required column missing from test data: {col}")


    states_scaled = scaler.transform(df_test[feature_cols].values)
    y_true_all = df_test["Decision_Label"].values.astype(int)
    class_names_all = (df_test["Class_Index"].astype(str).str.replace(r"_\d+$", "", regex=True).values)

    ppo_agent = PPOActorCritic(state_dim=len(feature_cols),action_dim=2).to(device)
    ppo_agent.load_state_dict(torch.load(ppo_model_path,map_location=device))
    ppo_agent.eval()

    reward_model = RewardModel(state_dim=len(feature_cols)).to(device)
    reward_model.load_state_dict(torch.load(rm_model_path,map_location=device))
    reward_model.eval()

    cf_module = CognitiveCounterfactualModule(
        scm_params_path=scm_params_path,
        action0_risk_weight=1.0,
        action1_risk_weight=0.4,
        margin_threshold_action0=0.70,
        margin_threshold_action1=0.30,
        lr=0.02,
        max_steps=6,
        early_stop_patience=2,
        boundary_refine_steps = 10)

    warmup_state = torch.tensor(states_scaled[0], dtype=torch.float32,device=device)

    warmup_action, _, _, _ = (cf_module.evaluate_policy_decision(ppo_actor=ppo_agent,single_state=warmup_state))
    _ = cf_module.optimize_counterfactual(ppo_actor=ppo_agent,single_state=warmup_state,policy_action=warmup_action)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(" CF warm-up completed")
    print(f" Starting PPO model evaluation "
        f"(Test samples: {len(df_test)} | "
        f"Device: {device})...")

    y_pred_all = []
    cf_triggered_all = []
    cf_valids_all = []
    d_cfs_all = []
    vulnerabilities_all = []
    cf_penalties_all = []
    cf_times_ms_all = []
    policy_confidences_all = []
    policy_margins_all = []
    rm_preference_gaps_all = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_start_time = time.perf_counter()

    for i in range(len(df_test)):
        state_tensor = torch.tensor(
            states_scaled[i],
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            policy_probs = (ppo_agent.get_action_probs(state_tensor.unsqueeze(0)).squeeze(0))
        policy_action = torch.argmax(policy_probs).item()
        policy_confidence = (policy_probs[policy_action].item())
        policy_margin = (torch.abs(policy_probs[1] - policy_probs[0]).item())

        pred_action = policy_action

        with torch.no_grad():
            action0_tensor = torch.tensor([0.0],dtype=torch.float32,device=device)
            action1_tensor = torch.tensor([1.0],dtype=torch.float32,device=device)
            rm_reward_action0 = reward_model.compute_action_reward(state=state_tensor.unsqueeze(0),action=action0_tensor,temperature=1.0).item()
            rm_reward_action1 = reward_model.compute_action_reward(state=state_tensor.unsqueeze(0),action=action1_tensor,temperature=1.0).item()

            rm_preference_gap = (rm_reward_action1- rm_reward_action0)

        cf_triggered = cf_module.should_trigger_cf(
            policy_action=policy_action,
            policy_margin=policy_margin,
            rm_preference_gap=rm_preference_gap
        )

        is_valid_cf = False
        d_cf = float("nan")
        vulnerability = float("nan")
        cf_penalty = 0.0
        cf_time_ms = 0.0

        if cf_triggered:

            if device.type == "cuda":
                torch.cuda.synchronize()
            cf_start_time = time.perf_counter()

            (d_cf,is_valid_cf,best_s_cf) = cf_module.optimize_counterfactual(
                ppo_actor=ppo_agent,
                single_state=state_tensor,
                policy_action=policy_action
            )
            (cf_penalty,vulnerability) = (
                cf_module.compute_vulnerability_penalty(
                    policy_action=policy_action,
                    is_valid=is_valid_cf,
                    d_cf=d_cf
                )
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            cf_end_time = time.perf_counter()
            cf_time_ms = (cf_end_time- cf_start_time) * 1000.0

        y_pred_all.append(pred_action)
        cf_triggered_all.append(cf_triggered)
        cf_valids_all.append(is_valid_cf)
        d_cfs_all.append(d_cf)
        vulnerabilities_all.append(vulnerability)
        cf_penalties_all.append(cf_penalty)
        cf_times_ms_all.append(cf_time_ms)
        policy_confidences_all.append(policy_confidence)
        policy_margins_all.append(policy_margin)
        rm_preference_gaps_all.append(rm_preference_gap)


    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_end_time = time.perf_counter()
    total_inference_time_s = (inference_end_time - inference_start_time)
    mean_inference_time_ms = (total_inference_time_s * 1000.0 / len(df_test))

    y_pred_all = np.asarray(y_pred_all,dtype=int)
    cf_triggered_all = np.asarray(cf_triggered_all,dtype=bool)
    cf_valids_all = np.asarray(cf_valids_all, dtype=bool)
    d_cfs_all = np.asarray(d_cfs_all,dtype=np.float64)
    vulnerabilities_all = np.asarray(vulnerabilities_all,dtype=np.float64)
    cf_penalties_all = np.asarray(cf_penalties_all,dtype=np.float64)
    cf_times_ms_all = np.asarray(cf_times_ms_all,dtype=np.float64)
    policy_confidences_all = np.asarray(policy_confidences_all,dtype=np.float64)
    policy_margins_all = np.asarray(policy_margins_all,dtype=np.float64)
    rm_preference_gaps_all = np.asarray(rm_preference_gaps_all,dtype=np.float64)

    unique_classes = np.unique(class_names_all)
    class_reports = []
    for cls in unique_classes:
        class_mask = (class_names_all == cls)
        metrics = compute_evaluation_metrics(
            y_true=y_true_all[class_mask],
            y_pred=y_pred_all[class_mask],
            cf_triggered=cf_triggered_all[class_mask],
            cf_valids=cf_valids_all[class_mask],
            d_cfs=d_cfs_all[class_mask],
            vulnerabilities=vulnerabilities_all[class_mask],
            cf_penalties=cf_penalties_all[class_mask],
            cf_times_ms=cf_times_ms_all[class_mask]
        )
        metrics["Class"] = cls
        class_reports.append(metrics)

    df_class_report = pd.DataFrame(class_reports)
    cols = (["Class"]+ [col for col in df_class_report.columns if col != "Class"])
    df_class_report = df_class_report[cols]

    overall_metrics = (
        compute_evaluation_metrics(
            y_true=y_true_all,
            y_pred=y_pred_all,

            cf_triggered=cf_triggered_all,
            cf_valids=cf_valids_all,
            d_cfs=d_cfs_all,

            vulnerabilities=vulnerabilities_all,

            cf_penalties=cf_penalties_all,

            cf_times_ms=cf_times_ms_all
        )
    )

    # =========================
    # Print only the per-class and overall result tables
    # =========================
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print("\n" + "=" * 160)
    print("Per-Class Test Results")
    print("=" * 160)
    print(
        df_class_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN"
        )
    )

    overall_report = pd.DataFrame([overall_metrics])
    overall_report["Total Inference Time(s)"] = total_inference_time_s
    overall_report["Mean Inference Time(ms/sample)"] = mean_inference_time_ms

    print("\n" + "=" * 160)
    print("Overall Test Results")
    print("=" * 160)
    print(
        overall_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN"
        )
    )



if __name__ == "__main__":
    evaluate_ppo_model()