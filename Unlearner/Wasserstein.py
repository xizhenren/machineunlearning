# Unlearner/Wasserstein.py
import numpy as np

"""
def wdp_epsilon_from_params(theta_unlearn: np.ndarray,
                            theta_retrain: np.ndarray,
                            mu: float = 1.0) -> float:
    
    # Wasserstein 认证遗忘的 epsilon：当两个输出分布是同噪声平移时，
    # W_mu 距离等于参数差的 L2 范数（与 mu 无关）。
    
    diff = theta_unlearn.reshape(-1) - theta_retrain.reshape(-1)
    return float(np.linalg.norm(diff, ord=2))

    
def wdp_sigma_from_sensitivity(delta2: float, epsilon_target: float, mu: float = 1.0, delta: float = 1e-5) -> float:
    
    #使用 Generalized (µ, ε)-WDP 标定噪声 σ
    #- delta2 是灵敏度估计
    #- epsilon_target 是目标隐私预算
    #- mu 是 Wasserstein 阶数，默认 1
    #- delta 是松弛项（failure probability）
    
    assert epsilon_target > 0 and mu >= 1.0
    # delta 的作用：在标定时容忍一定的失败概率
    # WDP 通常用标准的公式进行标定，delta 会影响隐私预算的容忍范围
    sigma = delta2 / (2 * epsilon_target)**mu
    return sigma
"""



def wdp_sigma_from_sensitivity(delta2: float, epsilon_target: float, mu: float = 2.0, delta: float = 1e-5) -> float:
    
    #使用 Generalized (µ, ε)-WDP 标定噪声 σ
    #- delta2 是灵敏度估计
    #- epsilon_target 是目标隐私预算
    #- mu 是 Wasserstein 阶数，默认 1
    #- delta 是松弛项（failure probability）
    
    assert epsilon_target > 0 and mu >= 1.0
    # delta 的作用：在标定时容忍一定的失败概率
    # WDP 通常用标准的公式进行标定，delta 会影响隐私预算的容忍范围
    return float(delta2 / ((2.0 * epsilon_target) ** mu))


def wdp_epsilon_from_params(theta_unlearn: np.ndarray,
                            theta_retrain: np.ndarray,
                            mu: float = 1.0) -> float:
    
    # Wasserstein 认证遗忘的 epsilon：当两个输出分布是同噪声平移时，
    # W_mu 距离等于参数差的 L2 范数（与 mu 无关）。
    
    diff = theta_unlearn.reshape(-1) - theta_retrain.reshape(-1)
    return float(np.linalg.norm(diff, ord=2))