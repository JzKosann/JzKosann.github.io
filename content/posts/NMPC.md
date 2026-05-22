---
title: "NMPC Controller for Autonomous Drone Navigation"
date: 2025-05-30T22:57:11+08:00
tags:
  - Nonlinear Model Predictive Control
  - acados
  - Flightmare
math: true
description: "Derivation of the MPC quadratic programming formulation — cost function, prediction horizon, constraint handling — and a demonstration running an NMPC controller in the Flightmare simulator."
---

## Quadratic Optimization: The Core Idea

At its heart, quadratic optimization seeks to minimize a quadratic objective function subject to constraints.

In its general form:

$$
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_n
\end{bmatrix}^T
\begin{bmatrix}
q_1 & 0 & \cdots & 0\\
0 & q_2 & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & q_n
\end{bmatrix}
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_n
\end{bmatrix} = q_1z_1^2 + q_2z_2^2 + \dots + q_nz_n^2
\tag{1}
$$

This structure forms the backbone of many optimization problems, including those in Model Predictive Control (MPC).

---

## Building the MPC Quadratic Programming Model

Consider a dynamic system governed by:

$$
x(k+1) = Ax(k) + Bu(k)
$$

To implement MPC, we define the future states and inputs over a prediction horizon.

At time $k$, the stacked state vector is:

$$
X_k =
\begin{bmatrix}
x(k|k)\\
x(k+1|k)\\
\vdots\\
x(k+N|k)
\end{bmatrix}
\tag{2}
$$

The stacked control input vector is:

$$
U_k =
\begin{bmatrix}
u(k|k)\\
u(k+1|k)\\
\vdots\\
u(k+N-1|k)
\end{bmatrix}
\tag{3}
$$

Here, $N$ represents the prediction horizon.

---

## Cost Function

The cost function penalizes both state error and control effort over the horizon:

$$
J=
\sum_{i=0}^{N-1}
\left(
E(k+i|k)^T Q\, E(k+i|k)
+
u(k+i|k)^T R\, u(k+i|k)
\right)
+
x(k+N)^T F\, x(k+N)
\tag{4}
$$

When the reference is zero, $E = x$, so the cost simplifies to:

$$
J=
\sum_{i=0}^{N-1}
\left(
x(k+i|k)^T Q\, x(k+i|k)
+
u(k+i|k)^T R\, u(k+i|k)
\right)
+
x(k+N)^T F\, x(k+N)
\tag{5}
$$

> The term $x(k+N)^T F\, x(k+N)$ is the **terminal cost**, which ensures stability at the end of the horizon.

### Error Weight Matrix $Q$

$Q$ penalizes state deviation. Larger diagonal entries impose stronger corrections:

$$
Q=
\begin{bmatrix}
q_1 & & \\
& \ddots & \\
& & q_N
\end{bmatrix}
\tag{6}
$$

### Control Weight Matrix $R$

$R$ penalizes control effort. Larger diagonal entries produce smoother inputs:

$$
R=
\begin{bmatrix}
r_1 & & \\
& \ddots & \\
& & r_N
\end{bmatrix}
\tag{7}
$$

---

## Derivation: Compact QP Form

### Predicting Future States

Starting from $x(k+1) = Ax(k) + Bu(k)$, unrolling over the horizon gives:

$$
\begin{cases}
x(k|k)   = x(k)\\
x(k+1|k) = Ax(k) + Bu(k|k)\\
x(k+2|k) = A^2x(k) + ABu(k|k) + Bu(k+1|k)\\
\quad\vdots\\
x(k+N|k) = A^N x(k) + \cdots + A^{N-1}Bu(k|k)
\end{cases}
\tag{8}
$$

Define the propagation matrix $M$ and control influence matrix $C$:

$$
M=
\begin{bmatrix}
I\\ A\\ A^2\\ \vdots\\ A^N
\end{bmatrix},
\qquad
C=
\begin{bmatrix}
0 & 0 & \cdots & 0\\
B & 0 & \cdots & 0\\
AB & B & \cdots & 0\\
A^2B & AB & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
A^{N-1}B & A^{N-2}B & \cdots & B
\end{bmatrix}
$$

Then the full state trajectory is:

$$
X(k) = M\,x(k) + C\,U(k)
\tag{9}
$$

### Computing the Compact Cost

Substituting Eq. (9) into the cost function yields the standard QP form:

$$
J =
X(k)^T \mathbf{Q}\, X(k)
+
U(k)^T \mathbf{R}\, U(k)
\tag{10}
$$

Expanding in terms of $x(k)$ and $U(k)$:

$$
\begin{cases}
J = x(k)^T G\, x(k) + 2\,x(k)^T E\, U(k) + U(k)^T H\, U(k)\\[4pt]
G = M^T \mathbf{Q} M\\
E = M^T \mathbf{Q} C\\
H = C^T \mathbf{Q} C + \mathbf{R}
\end{cases}
\tag{11}
$$

Since $x(k)$ is known at each time step, minimizing $J$ reduces to minimizing over $U(k)$ only.

---

## Constraint Functions

### Equality Constraints

For a QP problem of the form:

$$
\min_u \quad \frac{1}{2}u^T H\, u + u^T f
\tag{12}
$$

subject to $M_{eq}\, u = b_{eq}$, the Lagrangian is:

$$
J_L =
\frac{1}{2}u^T H\, u
+ u^T f
+ \lambda^T(M_{eq}\, u - b_{eq})
\tag{14}
$$

Setting the KKT conditions to zero:

$$
\begin{aligned}
\frac{\partial J_L}{\partial u}      &= Hu + f + M_{eq}^T \lambda = 0 \\[4pt]
\frac{\partial J_L}{\partial \lambda} &= M_{eq}\, u - b_{eq}       = 0
\end{aligned}
\tag{15}
$$

This gives the linear system:

$$
\begin{bmatrix}
H & M_{eq}^T\\
M_{eq} & 0
\end{bmatrix}
\begin{bmatrix}
u\\ \lambda
\end{bmatrix} =
\begin{bmatrix}
-f\\ b_{eq}
\end{bmatrix}
\tag{16}
$$

with closed-form solution:

$$
\begin{bmatrix}
u^*\\ \lambda^*
\end{bmatrix} =
\begin{bmatrix}
H & M_{eq}^T\\
M_{eq} & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
-f\\ b_{eq}
\end{bmatrix}
\tag{17}
$$

### Inequality Constraints

When inequality constraints are present, closed-form solutions no longer exist. In practice these are handled by:

- Active-set or interior-point QP solvers
- Dedicated solvers such as **acados** (used in this work)
- Commercial optimization packages (OSQP, GUROBI, etc.)

---

## Control in Simulator

YouTube video 👇

<a href="https://youtu.be/0tH68vLpp1M">
  <img src="/NMPC_img/video_index.png" width="700">
</a>
