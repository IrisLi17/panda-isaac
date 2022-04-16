from isaacgym.torch_utils import quat_conjugate, quat_mul
import torch
import numpy as np


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(dpose, j_eef, damping):
    num_envs = dpose.shape[0]
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=dpose.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

def control_osc(dpose, kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, dof_pos, dof_vel, hand_vel):
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=dpose.device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)

def pseudo_inverse(M, damped=True):
    lam = 0.2 if damped else 0.0
    U, S, VH = torch.linalg.svd(M)
    new_S = torch.zeros_like(M)
    for i in range(S.shape[1]):
        new_S[:, i, i] = S[:, i] / (S[:, i] * S[:, i] + lam * lam)
    return torch.transpose(VH, 1, 2) @ torch.transpose(new_S, 1, 2) @ torch.transpose(U, 1, 2)

def control_cartesian_impedance(dpose: torch.tensor, kp: torch.tensor, kd: torch.tensor, kp_null: float, kd_null: float, 
                                default_dof_pos_tensor: torch.tensor, j_eef: torch.tensor, dof_pos: torch.tensor, dof_vel: torch.tensor):
    j_eef_T = torch.transpose(j_eef, 1, 2)
    j_T_pinv = pseudo_inverse(j_eef_T)
    tau_task = j_eef_T @ (kp @ dpose - kd @ (j_eef @ dof_vel))
    tau_nullspace = (torch.eye(7, device=dpose.device).unsqueeze(dim=0).repeat(dpose.shape[0], 1, 1) - j_eef_T @ j_T_pinv) @ \
        (kp_null * (default_dof_pos_tensor.view(1, -1, 1) - dof_pos) - kd_null * dof_vel)
    tau_d = tau_task + tau_nullspace
    # Should add coriolis, for now disable gravity to bypass it
    # Should saturateTorque
    return tau_d.squeeze(-1)

#   // Cartesian PD control with damping ratio = 1
#   tau_task << jacobian.transpose() *
#                   (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));
#   // nullspace PD control with damping ratio = 1
#   tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
#                     jacobian.transpose() * jacobian_transpose_pinv) *
#                        (nullspace_stiffness_ * (q_d_nullspace_ - q) -
#                         (2.0 * sqrt(nullspace_stiffness_)) * dq);
#   // Desired torque
#   tau_d << tau_task + tau_nullspace + coriolis;
#   // Saturate torque rate to avoid discontinuities
#   tau_d << saturateTorqueRate(tau_d, tau_J_d);