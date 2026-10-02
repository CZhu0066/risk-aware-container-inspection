# Evidence aggregation
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_hierarchical_logop_mixture(
        input_path: str,
        output_path: str,
        expert_weights_base: np.ndarray,
        expert_bias_base: np.ndarray,
        expert_confidence: np.ndarray = None,
        noise_std: float = 0.1,
        temperature: float = 1.0,  # Temperature scaling factor used to smooth the LLR and prevent overconfidence
        seed: int = 42
) -> pd.DataFrame:
    """
    Hierarchical Bayesian mixture model based on the Logarithmic Opinion Pool (LogOP):
    1. Independent inference layer (LogOP): add the prior log-odds to each expert's dynamic LLR (feature-weighted score), then convert it into an individual posterior using the Sigmoid function.
    2. Mixture consensus layer: use expert confidence priors to compute the Bayesian consensus posterior mean and epistemic disagreement of the group.
    Note: H=1 indicates that manual reinspection is required, while H=0 indicates that manual reinspection is not required.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Specified input file not found: {input_path}")

    np.random.seed(seed)
    df = pd.read_csv(input_path)

    feature_cols = [
        'Max_Heatmap_Value',
        'Anomaly_Probability',
        'Mean_Anomaly_Score',
        'Anomaly_Ratio_0.25',
        'Boundary_Transition_Entropy',
        'Gradient_Strength_in_Color_Band_Direction'
    ]
    meta_cols = ['Class_Index', 'Image_Path', 'Decision_Label']

    for col in meta_cols + feature_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing from the dataset: {col}")

    S_raw = df[feature_cols].values  # Shape: (N, 6)
    y = df['Decision_Label'].values  # Used only to compute the global prior
    N = len(df)
    d = len(feature_cols)
    n_experts = len(expert_weights_base)

    if expert_confidence is None:
        expert_confidence = np.ones(n_experts) / n_experts
    else:
        expert_confidence = np.array(expert_confidence) / np.sum(expert_confidence)

    # 1. Feature standardization (Z-score)
    scaler = StandardScaler()
    S_scaled = scaler.fit_transform(S_raw)

    # ==========================================
    # Top-level prior: compute the global prior log-odds
    # ==========================================
    p_h1_base = np.mean(y)
    # Clip extreme values to prevent -inf / inf in logarithmic operations
    p_h1_base = np.clip(p_h1_base, 1e-5, 1 - 1e-5)

    # Bayesian log-odds baseline intercept = ln(P(H=1) / P(H=0))
    prior_log_odds = np.log(p_h1_base / (1.0 - p_h1_base))

    # ==========================================
    # Independent inference layer: compute log-likelihood ratios (LLRs) and individual posteriors
    # ==========================================
    # Dynamically inject sample- and expert-specific perturbations into weights and biases
    weight_noise = np.random.normal(0, noise_std, size=(N, n_experts, d))
    W_dynamic = expert_weights_base[np.newaxis, :, :] + weight_noise  # (N, M, d)

    bias_noise = np.random.normal(0, noise_std, size=(N, n_experts))
    B_dynamic = expert_bias_base[np.newaxis, :] + bias_noise  # (N, M)

    # 1. Compute the log-likelihood ratio (LLR_m)
    # LLR = W^T * E + b
    # S_scaled[:, np.newaxis, :] shape: (N, 1, d) -> broadcast multiplication and sum over feature dimension d -> (N, M)
    LLR_dynamic = np.sum(S_scaled[:, np.newaxis, :] * W_dynamic, axis=2) + B_dynamic

    # Apply temperature scaling to avoid extreme probabilities of 1 or 0
    LLR_scaled = LLR_dynamic / temperature

    # 2. Combine the prior and likelihood to obtain posterior log-odds
    posterior_log_odds = prior_log_odds + LLR_scaled  # Shape: (N, M)

    # Clip to [-250, 250] to prevent np.exp overflow
    posterior_log_odds = np.clip(posterior_log_odds, -250, 250)

    # 3. Convert log-odds back to probability space P(H=1|E) using the Sigmoid function
    expert_posteriors = 1.0 / (1.0 + np.exp(-posterior_log_odds))  # Shape: (N, M)

    # ==========================================
    # Mixture consensus layer (Bayesian Mixture Consensus)
    # ==========================================
    # 1. Bayesian consensus posterior mean
    consensus_posterior = np.sum(expert_posteriors * expert_confidence, axis=1)

    # 2. Epistemic uncertainty / expert disagreement (weighted variance of the mixture distribution)
    diff_sq = (expert_posteriors - consensus_posterior[:, np.newaxis]) ** 2
    epistemic_uncertainty = np.sum(diff_sq * expert_confidence, axis=1)

    # ==========================================
    # Sample-level feature preference attribution (absolute contribution extraction)
    # ==========================================
    # In the linear model, the absolute feature influence is |W_j * E_j|
    mean_W_sample = np.average(W_dynamic, axis=1, weights=expert_confidence)  # (N, d)
    sample_impact = np.abs(S_scaled * mean_W_sample)

    sample_weights = sample_impact / (np.sum(sample_impact, axis=1, keepdims=True) + 1e-8)
    sample_weights_str = [
        np.array2string(np.round(w_row, 4), precision=4, separator=', ')
        for w_row in sample_weights
    ]

    # ==========================================
    # Save results
    # ==========================================
    df['posterior_prob'] = np.round(consensus_posterior, 4)
    df['expert_disagreement'] = np.round(epistemic_uncertainty, 6)
    df['average_weight_preference'] = sample_weights_str

    target_columns = [
        'Class_Index', 'Image_Path', 'Max_Heatmap_Value', 'Anomaly_Probability',
        'Mean_Anomaly_Score', 'Anomaly_Ratio_0.25', 'Boundary_Transition_Entropy',
        'Gradient_Strength_in_Color_Band_Direction', 'Decision_Label',
        'posterior_prob', 'expert_disagreement', 'average_weight_preference'
    ]
    df_output = df[target_columns]

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_output.to_csv(output_path, index=False)

    print("=" * 60)
    print("LogOP hierarchical Bayesian mixture model processing completed!")
    print(f"- Temperature scaling factor (temperature): {temperature}")
    print(f"- Base prior log-odds (prior_log_odds): {prior_log_odds:.4f}")
    print(f"- Mean consensus probability (posterior_prob): {df_output['posterior_prob'].mean():.4f}")
    print(f"- Proportion of extreme values (0 or 1): {((df_output['posterior_prob'] >= 0.999) | (df_output['posterior_prob'] <= 0.001)).mean() * 100:.2f}%")
    print(f"- Mean epistemic uncertainty (expert_disagreement): {df_output['expert_disagreement'].mean():.6f}")
    print("=" * 60)

    return df_output


if __name__ == "__main__":
    INPUT_PATH = "E:\\RLHF\\202606_results_revision\\ground_truth\\Dreal_merged.csv"
    OUTPUT_PATH = "E:\\RLHF\\20260804\\dataset\\Dreal_PPO.csv"

    # In the LogOP scheme, the regression parameters include Weight (W) and Bias (b)
    EXPERT_WEIGHTS_BASE = np.array([
        [0.85, 1.20, 0.90, 0.50, 0.30, 0.40],
        [1.50, 0.80, 1.10, 0.20, 0.10, 0.20],
        [0.40, 1.50, 0.60, 0.80, 0.60, 0.70],
        [1.00, 1.00, 1.00, 0.50, 0.50, 0.50],
        [0.50, 0.90, 1.20, 0.60, 0.40, 0.80]
    ])
    EXPERT_BIAS_BASE = np.array([0.10, -0.20, 0.05, 0.00, -0.15])

    # Expert confidence weights Alpha
    EXPERT_CONFIDENCE = np.array([0.2, 0.2, 0.2, 0.20, 0.2])

    processed_df = process_hierarchical_logop_mixture(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        expert_weights_base=EXPERT_WEIGHTS_BASE,
        expert_bias_base=EXPERT_BIAS_BASE,
        expert_confidence=EXPERT_CONFIDENCE,
        noise_std=0.15,
        temperature=2.0  # Increase this value (e.g., 2.0-5.0) to smooth scores if too many posterior probabilities are extreme
    )

    if processed_df is not None:
        print("\nPreview of the first 3 rows of generated results:")
        print(processed_df.head(3))