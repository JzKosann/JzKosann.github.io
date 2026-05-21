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

# Quadratic Optimization: The Core Idea

At its heart, quadratic optimization seeks to minimize a quadratic objective function.

In its general form:

$$
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_n
\end{bmatrix} ^ T
\begin{bmatrix}
q_1 & 0 & 0 &\\
0 & q_2 & 0 & 0 \\
0 & 0 &\ddots & 0\\
0 & 0 & 0 & q_n
\end{bmatrix}
\begin{bmatrix}
z_1\\
z_2\\
\vdots\\
z_n
\end{bmatrix}
=
q_1z_1^2 + q_2z_2^2 + \dots + q_nz_n^2
\tag{1}
$$

This structure forms the backbone of many optimization problems, including those in Model Predictive Control (MPC).

---

# Building the MPC Quadratic Programming Model

Consider a dynamic system governed by:

$$
x(k+1)=Ax(k)+Bu(k)
$$

To implement MPC, we define the future states and inputs over a prediction horizon.

At time \(k\), the stacked state vector is:

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

Here, \(N\) represents the prediction horizon.

---

# Cost Function

The cost function is defined as:

$$
J=
\sum_{i=0}^{N-1}
\left(
E(k+i|k)^TQE(k+i|k)
+
u(k+i|k)^TRu(k+i|k)
\right)
+
x(k+N)^TFx(k+N)
\tag{4}
$$

If the reference is zero, then:

$$
E = x
$$

Thus, the cost function becomes:

$$
J=
\sum_{i=0}^{N-1}
\left(
x(k+i|k)^TQx(k+i|k)
+
u(k+i|k)^TRu(k+i|k)
\right)
+
x(k+N)^TFx(k+N)
\tag{5}
$$

---

## Error Weight Matrix \(Q\)

The matrix \(Q\) penalizes state error:

$$
Q=
\begin{bmatrix}
q_1 &&\\
&\ddots &\\
&& q_N
\end{bmatrix}
\tag{6}
$$

Larger values indicate stronger penalties on state deviation.

---

## Input Weight Matrix \(R\)

The matrix \(R\) penalizes control effort:

$$
R=
\begin{bmatrix}
r_1 &&\\
&\ddots &\\
&& r_N
\end{bmatrix}
\tag{7}
$$

Larger values lead to smoother control actions.

---

> The term \(x(k+N)^TFx(k+N)\) is called the terminal cost.

---

# Derivation: From States to Optimization

## Predicting Future States

Starting from:

$$
x(k+1)=Ax(k)+Bu(k)
$$

The future states can be expanded as:

$$
\begin{cases}
x(k|k)=x(k)\\
x(k+1|k)=Ax(k|k)+Bu(k|k)\\
x(k+2|k)=A^2x(k|k)+ABu(k|k)+Bu(k+1|k)\\
\vdots\\
x(k+N|k)=A^Nx(k|k)+\cdots+A^{N-1}Bu(k|k)
\end{cases}
\tag{8}
$$

Define:

$$
M=
\begin{bmatrix}
I\\
A\\
A^2\\
\vdots\\
A^N
\end{bmatrix}
$$

and

$$
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

Then:

$$
X(k)=Mx(k)+CU(k)
\tag{9}
$$

---

# Computing the Cost Function

Substituting Equation (9) into the cost function:

$$
J=
X(k)^T\mathbf{Q}X(k)
+
U(k)^T\mathbf{R}U(k)
\tag{10}
$$

The optimization problem becomes:

$$
\begin{cases}
J=
x(k)^TGx(k)
+
2x(k)^TEU(k)
+
U(k)^THU(k)
\\
G=M^T\mathbf{Q}M
\\
E=M^T\mathbf{Q}C
\\
H=C^T\mathbf{Q}C+\mathbf{R}
\end{cases}
\tag{11}
$$

---

# Constraint Functions

## Equality Constraints

Consider the quadratic objective:

$$
J=
\frac{1}{2}u^THu + u^Tf
\tag{12}
$$

subject to:

$$
M_{eq}u=b_{eq}
\tag{13}
$$

Using Lagrange multipliers:

$$
J_L=
\frac{1}{2}u^THu
+
u^Tf
+
\lambda^T(M_{eq}u-b_{eq})
\tag{14}
$$

Taking derivatives:

$$
\begin{aligned}
\frac{\partial J_L}{\partial u}
&=
Hu+f+M_{eq}^T\lambda
=
0
\\
\frac{\partial J_L}{\partial \lambda}
&=
M_{eq}u-b_{eq}
=
0
\end{aligned}
\tag{15}
$$

This leads to:

$$
\begin{bmatrix}
H & M_{eq}^T\\
M_{eq} & 0
\end{bmatrix}
\begin{bmatrix}
u\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
-f\\
b_{eq}
\end{bmatrix}
\tag{16}
$$

Finally:

$$
\begin{bmatrix}
u^*\\
\lambda^*
\end{bmatrix}
=
\begin{bmatrix}
H & M_{eq}^T\\
M_{eq} & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
-f\\
b_{eq}
\end{bmatrix}
\tag{17}
$$

---

# Inequality Constraints

For inequality constraints, analytical solutions become difficult.

In practice, these are solved using:

- Numerical optimization algorithms
- Quadratic programming solvers
- Commercial optimization software

---

# Control in Simulator

YouTube video 👇

<a href="https://youtu.be/0tH68vLpp1M">
  <img src="/NMPC_img/video_index.png" width="700">
</a>
