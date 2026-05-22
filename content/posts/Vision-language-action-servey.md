---
title: "Vision-Language-Action Models: An Action Tokenization Survey"
date: 2026-05-21T20:25:34+09:00
draft: false
math: true
tags:
  - VLA
  - Robotics
  - Survey
description: "Reading notes on the survey paper 「A Survey on Vision-Language-Action Models: An Action Tokenization Perspective」. Covers 8 categories of action tokens with key papers, advantages, and limitations."
---

> Based on: [A Survey on Vision-Language-Action Models: An Action Tokenization Perspective](https://arxiv.org/abs/2507.01925v1)

---

## Action Token 概述

**Action token** 是 VLA 模型的输出。截至 2026-05-13，主流输出类型被归纳为 **8 种**：

| # | Action Token 类型 | 核心形式 |
|---|---|---|
| 1 | Language Description | 自然语言动作序列 |
| 2 | Code | 可执行代码片段 |
| 3 | Affordance | 空间交互表达（关键点、边界框等）|
| 4 | Trajectory | 动态运动轨迹 |
| 5 | Goal State | 预测未来观测 |
| 6 | Latent Representation | 隐向量序列 |
| 7 | Raw Action | 直接控制指令 |
| 8 | Reasoning | 决策过程的自然语言描述 |

---

## 1. Language Description

**形式：** 自然语言表达，从高层抽象规划到底层确定性动作描述均可覆盖。

### Key Papers

- **[3D-VLA](https://github.com/UMass-Embodied-AGI/3D-VLA)**、**[RoboMamba](https://github.com/lmzpai/roboMamba.git)**
  — 将 3D 场景理解、空间布置与视觉可行性预测整合进规划循环。
- **[HiRobot](https://www.pi.website/research/hirobot)**、**[RT-H](https://rt-hierarchy.github.io/)**、**[NaVILA](https://github.com/AnjieCheng/NaVILA.git)**
  — 分层架构（hierarchical framework）：高层用 VLM 做规划，底层负责控制策略。
- **[$\pi$-0.5](https://domrigby.github.io/robotics/Pi0.5VLA.html)**
  — 将 HiRobot 的分层架构整合进单一 VLA 模型。

### 优势

1. 直接复用 LLM/VLM 的理解、推理与规划能力，表达能力强。
2. 可利用来自 web 的海量 co-training 数据，为 agent 提供经验知识。
3. 天然适合 **long-horizon 规划**（大语言模型本身就输出长序列）。
4. 高层语言描述带来良好的**可解释性**。

### 局限

1. 表达模糊，难以指定精细的底层控制行为。
2. 推理延迟高。

---

## 2. Code

**形式：** 可执行代码片段或伪代码，将动作编码为结构化程序逻辑。

---

## 3. Affordance

**形式：** 捕捉对象任务属性与交互能力的空间表达，包括关键点、边界框、affordance map 等，本质上描述的是「物体与机器人之间的动作可能性」。

利用 VLM 的空间解释能力，输入多模态信息后识别可产生动作的区域并评估可行性。同一高层感知结果可跨机器人系统复用，具有较强的**迁移泛化能力**。

Affordance 的四种表现形式：`keypoints`、`bounding boxes`、`segmentation masks`、`affordance map`。

---

### 3.1 Key Points

利用 VLM 提供精确的空间边界与交互目标（如对象把手、接触边缘）。

$$
\mathbf{k} = [\mathbf{x},\ \mathbf{d}], \quad \mathbf{x}, \mathbf{d} \in \mathbb{R}^3
$$

其中 $\mathbf{x}$ 为空间接触位置（Spatial Contact Position），$\mathbf{d}$ 为交互方向（Interaction Direction）。

**Key Papers**

- **[KITE](https://github.com/priyasundaresan/kite_keypoint_training.git)** — 预测任务相关的语义关键点，随后用附加条件的技能完成底层控制。
- **[RoboPoint](https://github.com/wentaoyuan/RoboPoint.git)** — 将机器人操作映射到图像空间的 2D 关键点。
- **[CoPa](https://github.com/HaoxuHuang/copa.git)** — 两阶段流程：先粗识别目标语义，再精定位可操作点。
- **[KUDA](https://github.com/StoreBlank/KUDA.git)** — 为目标物体打 keypoint 标记，辅以基于视觉和语言描述的脚本检索库，让 VLM 聚焦具体物体（vision prompting）。
- **[RAM](https://github.com/yxKryptonite/RAM_code.git)** — 训练数据使用 out-of-domain 数据（不含目标机器人底层动作标签，但含丰富物理常识），标记关键点后构建 2D vision data library，遇到新环境时检索相似场景作为参考。
- **[ReKep](https://github.com/huangwl18/ReKep.git)** — 将复杂任务映射为基于 3D Key Points 的数学约束，通过求解器完成机器人动作。
- **[OmniManip](https://omnimanip.github.io/)** — 将所有识别到的对象抽象为类似搭积木的表达，同时对物体坐标进行 keypoint 风格的标记，提升操作泛化性和可解释性。
- **[Magma](https://github.com/microsoft/Magma.git)**、**[VidBot](https://github.com/ethz-mrl/VidBot.git)** — 将静态 keypoint 扩展为 keypoint sequence，接近于下文的 Trajectory token。

---

### 3.2 Bounding Boxes

用矩形框标注目标区域，作为空间交互的粗粒度表达。

---

### 3.3 Segmentation Masks

用像素级掩码精确标注可交互区域，适合形状不规则的目标。

---

### 3.4 Affordance Map

在图像上生成连续热力图，显式标注每个像素的交互可能性分布。

---

## 4. Trajectory

**形式：** 完整操作轨迹，分为三类。

---

### 4.1 Point Trajectory

将动作编码为离散序列：在时间跨度 $T$ 内的 $K$ 个关键点，提供目标引导与精确数量控制。

**Key Papers**

- **[LLARVA](https://github.com/Dantong88/LLARVA.git)** — 通过 instruction tuning，输入 `robot mode`、`control mode`、`robot task`、`proprioceptive info`、`predicted steps`，输出动作轨迹文本并可视化。
- **[Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment.git)** — 大规模跨机器人具身数据集与基线模型。

---

### 4.2 Visual Trajectory

直接在像素空间 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$（或 $\mathbb{R}^{T \times H \times W \times 3}$）生成路径，输出为可视化图像或视频流。可解释性极强。

**Key Papers**

- **[ARM4R](https://github.com/Dantong88/arm4r.git)** — 生成 4D Representations，在 $T$ 个离散帧中表达三维坐标。
- **[Magma](https://github.com/microsoft/Magma.git)** — 生成 motion sequence 并可视化到图片/视频；同时提供 VLA 基础平台，可用于其他模型的学习。⭐

---

### 4.3 Optical Flow

更密集的运动表达，公式化为动作场（motion field）：

$$
\mathbf{F} \in \mathbb{R}^{H \times W \times 2}
$$

描述两帧之间的逐像素位移，可对复杂多目标交互进行显式建模。

**Key Papers**

- **[AVDC](https://github.com/flow-diffusion/AVDC.git)** — 用人类/机器人演示视频训练 diffusion model 生成 optical flow。计算代价较高且易产生幻觉（hallucination）。
- **[ATM](https://xingyu-lin.github.io/atm)** — 通过预测轨迹上任意点，缓解了 AVDC 的上述问题。
- **[Im2Flow2Act](https://im-flow-act.github.io/)** — 无需真实机器人数据，从人类演示视频学习；相比 ATM，仅关注目标物体的 flow。
- **[FLIP](https://nus-lins-lab.github.io/flipweb/)** — 结合视频构建的 world model 进行 model-based planning，同时以 flow 和 video plan 为条件预测动作。

---

## 5. Goal State

**形式：** 预测的未来观测值（图像、点云或视频片段），可视化地表达动作序列预期结果。

---

## 6. Latent Representation

**形式：** 有意训练的隐向量序列（latent vector sequence），其中编码了动作相关信息。不直接对应语言或像素，而是在压缩表示空间中传递控制信号。

---

## 7. Raw Action

**形式：** 直接的控制指令，如关节角度、末端执行器位姿、速度命令等，是最底层的 action token 形式。

---

## 8. Reasoning

**形式：** 自然语言表达，显式描述做决策的推理过程，以此引导特定动作的生成。与 Language Description 的区别在于：Reasoning 重在「为什么」，Language Description 重在「做什么」。
