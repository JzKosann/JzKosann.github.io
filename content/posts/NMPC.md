---
title: "Research on Autonomous Navigation Systems for Traversing Drones - NMPC Controller"
date: 2025-05-30T22:57:11+08:00
tags:
  - Nonlinear Model Predictive Control
  - acados
  - Flightmare
math: true
description: "Run NMPC Controller in Flightmare"
---

## Quadratic Optimization: The Core Idea

At its heart, quadratic optimization seeks to minimize a quadratic objective function. In its general form, it often appears as:
$$
\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}^T
\begin{bmatrix}q_1&&&\\&q_2&&\\&&\ddots&\\&&&q_n\end{bmatrix}
\begin{bmatrix}z_1\\z_2\\\vdots\\z_n\end{bmatrix}
=
q_1z_1^2+q_2z^2_2 + \dots + q_nz_n^2
\tag{1}
$$
This structure forms the backbone of many optimization problems, including those in Model Predictive Control (MPC).

## Building the MPC Quadratic Programming Model

Consider a dynamic system governed by the linear state-space equation: $x(k+1)=Ax(k)+Bu(k)$.
To implement Model Predictive Control, we define the system's states and inputs over a future prediction horizon.

At a given time $k$, and looking ahead $N$ prediction intervals, the system's states are stacked into a vector:
$$
X_k = \begin{bmatrix}x(k|k)\\x(k+1|k)\\...\\x(k+N|k)\end{bmatrix}
\tag{2}
$$
Similarly, the corresponding sequence of control inputs over the same horizon is:
$$
U_k=\begin{bmatrix}u(k|k)\\u(k+1|k)\\...\\u(k+N-1|k)\end{bmatrix}
\tag{3}
$$
Here, $N$ represents the **prediction horizon**, indicating how many future time steps we predict from the current time $k$.

The performance of the system is quantified by a **Cost Function**, typically a sum of quadratic terms that penalize deviations from desired states and excessive control effort:
$$
J=\sum^{N-1}_{i=0}(E(k+i|k)^TQE(k+i|k)+u(k+i|k)^TRu(k+i|k))
+x(k+N)^TFx(k+N)\\
\tag{4}
$$
In many control problems, the system's output $y$ is directly equivalent to its state $x$. If our reference $R$ is set to zero (i.e., we aim for the states to converge to zero), then the error $E$ simplifies to $E=y-R=x-0=x$.
Under this simplification, the Cost Function $J$ can be rewritten as:
$$
J=\sum^{N-1}_{i=0}(x(k+i|k)^TQx(k+i|k)+u(k+i|k)^TRu(k+i|k))
+x(k+N)^TFx(k+N)\\
\tag{5}
$$
The matrices $Q$, $R$, and $F$ play crucial roles in shaping the control behavior:

**Error Weight Matrix $Q$**: This diagonal matrix assigns weights to the state variables. A higher value $q_j$ in $Q$ signifies that the $j$-th state variable is more critical, and its deviation from the reference should be penalized more heavily.
$$
Q=\begin{bmatrix}q_1&&\\&\ddots&\\&&q_N\end{bmatrix}\\
\tag{6}
$$

**Input Weight Matrix $R$**: Similarly, this diagonal matrix penalizes the control inputs. A larger $r_j$ means that the $j$-th input should be applied more sparingly, promoting smoother control actions and preventing excessive actuator wear.
$$
R=\begin{bmatrix}r_1&&\\&\ddots&\\&&r_N\end{bmatrix}\\
\tag{7}
$$

> The term $x(k+N)^TFx(k+N)$ in Equation (4) is known as the **terminal error**. It penalizes the state at the very end of the prediction horizon, ensuring that the system is guided towards a desirable final state within the predicted window. For a deeper understanding of terminal weights, refer to: [【模型预测控制】笔记 （一）终端权重的由来 - 知乎](https://zhuanlan.zhihu.com/p/399207343).

## Derivation: From States to an Optimization Problem

### Expressing All Predicted States Using the Initial State and Control Inputs

Starting from the system's dynamics $x(k+1)=Ax(k)+Bu(k)$, and considering the stacked state $X_k$ and input $U_k$ vectors, we can project the future states:
$$
\begin{cases}
x(k|k)= x(k)\\
x(k+1|k)=Ax(k|k)+Bu(k|k)\\
x(k+2|k)=A^2x(k|k)+Bu(k+1|k)+ABu(k|k)\\
\vdots\\
x(k+N|k)=A^Nx(k|k)+Bu(k+N-1|k)+ABu(k+N-2|k)+\cdots+A^{N-1}Bu(k|k)\\
\end{cases}
\tag{8}
$$
These relationships can be compactly expressed in matrix form by defining $M$ and $C$ matrices:
$$
M=\begin{bmatrix}I\\A\\A^2\\...\\A^N\end{bmatrix} \quad \text{and} \quad C=\begin{bmatrix}0&0&0&\dots &0\\B&0&0&\cdots &0\\AB&B&0&\cdots &0\\A^2B&AB&B&\cdots &0\\\vdots &\vdots &\vdots &\ddots &\vdots \\A^{N-1}B&A^{N-2}B&\cdots&\cdots&B\end{bmatrix}
$$
With these matrices, the stacked future states $X(k)$ can be elegantly represented as a function of the current state $x(k)$ and the future control inputs $U(k)$:
$$
X(k)=Mx(k)+CU(k)\\
\tag{9}
$$

### Computing the Cost Function for All Predicted States

Substituting the expression for $X(k)$ from (9) into the cost function (5), we can transform $J$ into a quadratic form primarily dependent on the control inputs $U(k)$:
$$
J=X(k)^T\mathbf{Q}X(k)+U(k)^T\mathbf{R}U(k)\tag{10}
$$
Here, $\mathbf{Q}$ is a block diagonal matrix of size $(\text{number of state variables} \times \text{prediction horizon}) \times (\text{number of state variables} \times \text{prediction horizon})$, built from the individual $Q$ matrices and $F$ for the terminal state. Similarly, $\mathbf{R}$ is a block diagonal matrix of size $(\text{number of input variables} \times \text{prediction horizon}) \times (\text{number of input variables} \times \text{prediction horizon})$, constructed from the individual $R$ matrices.

Combining (9) and (10), the new cost function $J$ takes the standard quadratic programming form:
$$
\begin{cases}
J=x(k)^TGx(k)+2x(k)^TEU(k)+U(k)^THU(k)\\
G=M^T\mathbf{Q}M\\
E=M^T\mathbf{Q}C\\
H=C^T\mathbf{Q}C + \mathbf{R}\\
\end{cases}
\tag{11}
$$

## Constraint Functions: Navigating the Solution Space

In addition to the objective function, MPC problems often involve constraints on states and inputs. These can be categorized into equality and inequality constraints.

### Equality Constraints: Leveraging Lagrange Multipliers

For a quadratic programming problem with a performance index:
$$
J=\frac{1}{2} u^T H u + u^T f
\tag{12}
$$
Subject to linear equality constraints:
$$
M_{eq}u = b_{eq}
\tag{13}
$$
where $M_{eq}$ is an $m \times n$ matrix, and $b_{eq}$ is an $m \times 1$ vector.

To solve this problem, we introduce an $m \times 1$ vector of **Lagrange multipliers** $\lambda$. These multipliers allow us to incorporate the constraints directly into the objective function, forming an augmented Lagrangian:
$$J_L=\frac{1}{2}u^THu + u^Tf +\lambda ^T (M_{eq}u-b_{eq}) \tag{14}$$
To find the optimal solution, we take the partial derivatives of $J_L$ with respect to $u$ and $\lambda$, and set them to zero:
$$
\frac{\partial J_L}{\partial u} = Hu+f+M_{eq}^T\lambda = 0 \\
\frac{\partial J_L}{\partial \lambda} = M_{eq}u-b_{eq}=0
\tag{15}
$$
These equations can be concisely expressed in matrix form:
$$
\begin{bmatrix}H & M_{eq}^T \\ M_{eq}& 0 \end{bmatrix}
\begin{bmatrix}u\\\lambda \end{bmatrix}
=
\begin{bmatrix}-f\\b_{eq}\end{bmatrix}
\tag{16}
$$
Assuming the KKT (Karush-Kuhn-Tucker) matrix $\begin{bmatrix}H & M_{eq}^T \\ M_{eq}& 0 \end{bmatrix}$ from (16) is invertible, we can directly solve for the optimal control input $u^*$ and the Lagrange multipliers $\lambda^*$:
$$
\begin{bmatrix}u^*\\ \lambda^* \end{bmatrix}
=
\begin{bmatrix}H_{n \times n} & M_{eq_{n \times m}}^T \\ M_{eq_{m\times n}}& 0 \end{bmatrix}^{-1}
\begin{bmatrix}-f_{n\times 1}\\b_{eq_{m \times 1}}\end{bmatrix}
\tag{17}
$$

### Inequality Constraints: The Realm of Numerical Methods and Commercial Software

For problems involving inequality constraints, direct analytical solutions become significantly more complex. In practice, these are typically handled using:

* **Numerical Optimization Algorithms:** Specialized algorithms are designed to solve quadratic programs with inequality constraints.

## Control it in simulator
youtube video👇

[![video](/NMPC_img/video_index.png)](https://youtu.be/0tH68vLpp1M)
