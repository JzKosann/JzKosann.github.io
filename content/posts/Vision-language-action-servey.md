+++
date = '2026-05-21T20:25:34+09:00'
draft = false
title = 'Vision Language Action Servey (Updating)'
math = true
+++

All my research in this blog depend on paper [A Survey on Vision-Language-Action Models: An Action Tokenization Perspective](https://arxiv.org/abs/2507.01925v1)

## **Action token**

`action token` 是`VLA` 模型的输出，当前（2026-5-13）主流的输出类型被归类成8种：

### **Language Description**
自然语言表达，描述的是动作序列，涵盖高层抽象语言计划到底层确定性语言动作。
    
#### Papers
    
- [3D-VLA](https://github.com/UMass-Embodied-AGI/3D-VLA)、[RoboMamba](https://github.com/lmzpai/roboMamba.git)
将3D场景理解、空间布置和视觉可行性预测结果合并进规划循环中。
- [HiRobot](https://www.pi.website/research/hirobot)、[RT-H](https://rt-hierarchy.github.io/)、[NaVILA](https://github.com/AnjieCheng/NaVILA.git)
`hierarchical framework` ，高层负责`VLM` ，底层负责控制策略。
- [$\pi$-0.5](https://domrigby.github.io/robotics/Pi0.5VLA.html)
将`HiRobot`的分层架构整合进一个VLA模型中。
    
#### Advantages
    
1. 源于LLMs、VLMs的理解、解释和规划能力，并且高层模型输出的本身就是自然语言，所以他的表达能力很好，`Language Description as Action Tokens` 刚好在属性上对其了模型本身输出的结果。
2. `co-training data` 的优势，因为大语言模型本身就采集来自web的一些文字和图片，在高层做决策的时候可以通过海量数据作为经验知识来帮助agent更好的决策。
3. 和第一点有关，因为大语言模型本身就是输出`long-horizon` 的结果，所以这类`action tokens` 很适合长规划。
4. 高层的模型输出的语言描述可以直接带来强**可解释性**

#### Demerits

1. 表达模糊，不足以指定更加精细的控制行为
2. 高延迟

### **Code**
    
一段可执行代码片段或者伪代码
    
### **Affordance**
    
捕捉了对象的任务特殊性质和交互属性的空间表达，通常是一些关键点、边界框、affordance map ，或者说是物体与机器人之间的`动作可能性`

*利用`VLM` 的空间解释能力，输入多模态信息，识别可以产生动作的区域和评价动作可行性。
具有强大的交叉泛化能力，同一个高层感知解释可以在不同的机器人系统中使用。 
显式地对任务相关的交互信息（例如抓取点或者操作平面）进行编码，让这些以物体为目标的操作变得更加高效。*

`Affordance` 可以表现的四种形式：`keypoints` 、`bounding boxes` 、`segmentation masks` 、`affordance map` 。

#### **Key Points**
    
利用`VLM` 提供准确的空间边界，提供交互目标（例如对象句柄或者接触边缘）的精确表达。 $\mathbf{k} =[\mathbf{x}, \space \mathbf{d}]$，这里 $\mathbf{x, d}\in \mathbb{R}^3$， $\mathbf{x}$表达的是空间接触位置`Spatial Contact Position` ， $\mathbf{d}$表达的是交互方向`Interaction Direction` 。

**Key Papers** 

[KITE:pointnet2_primitives](https://github.com/priyasundaresan/pointnet2_primitives.git)、[KITE:keypoint_training](https://github.com/priyasundaresan/kite_keypoint_training.git)、[KITE:semantic_grasping](https://github.com/priyasundaresan/kite_semantic_grasping.git)预测任务相关的关键点（语义对象点），随后在附加条件的技能中获取底层控制。

[RoboPoint](https://github.com/wentaoyuan/RoboPoint.git)

[CoPa](https://github.com/HaoxuHuang/copa.git) 是两个过程，先粗识别目标语义，选取最适合的以后再细识别可操作点。

[KUDA](https://github.com/StoreBlank/KUDA.git)利用给目标物体打标记的方式，给每个目标打上`keypoint` ，让`llm`或者`vlm` 聚焦到具体物体上，提供`vision prompting` 。除此之外KUDA还做了一个可用于检索的脚本库，根据视觉和语言描述，检索出相似的脚本给控制器。

[RAM](https://github.com/yxKryptonite/RAM_code.git)的特别的点应该在于训练数据上，RAM采用`out-of-domain` 数据，这一类数据不包含目标机器人的底层动作标签，但因为具备丰富的物理常识，具有较高的泛化性。RAM和KUDA有点相似，RAM将这类`out-of-domain` 数据标记关键点后，形成一个`(2D) vision data library` ，在面对新环境后开启检索，得出相似环境后，按照参考来进行动作。

[ReKep](https://github.com/huangwl18/ReKep.git)将复杂的任务映射到基于`3D Key Points` 的数学约束中，通过求解器来完成机器人的动作。

[OmniManip](https://omnimanip.github.io/)以视觉识别到的东西为中心，做个对比，普通的视觉采集的像素都是以场景为主要目标，因此无法具体识别物体的具体物理属性和环境交互能力，OmniManip将识别到的一切抽象成类似于搭积木的过程，所有物体共同构建出来一个场景，具有较强的机器人操作泛化和可解释性。这里不仅对物体有了标记，并且对物体的坐标也做了类似于`Keypoint` 的标记。

[Magma](https://github.com/microsoft/Magma.git)、[VidBot](https://github.com/ethz-mrl/VidBot.git)将静态的`keypoint`延伸到`keypoint sequence`里面，类似于`trajectory as action token` ，具备较强的运动规划能力。
    
- **`Bounding Boxes`**
- **`Segmentation Masks`**
- **`Affordance Map`**
    
### **Trajectory**

动态轨迹，完整的操作轨迹，可以分为三种

#### Point Trajectory

将动作编码为一个离散序列， $K$ critical points over a time span $T$，该序列提供目标和准确数量的引导。

##### Key papers

[LLARVA](https://github.com/Dantong88/LLARVA.git)使用的`instruction tuning` ，通过输入一个包含`robot mode`、`control mode`、`robot task`、`proprioceptive information`和`numbers of predicted steps` ，输出`动作轨迹.txt`，并将轨迹可视化。

[Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment.git)

#### Visual Trajectory

直接在像素空间(pixel space $\mathbf{I}\in \mathbb{R}^{H\times W \times 3}\space \text{or}\space \mathbf{I}\in \mathbb{R} ^ {T\times H \times W\times 3}$)生成一条路径。输出的是一个可以将预期动作视觉描绘出来的图片或者视频。可以通过布置点序列到观测帧上或者生成视频流让可视化曲线实体化。可解释性给到夯爆了。

##### Key Papers 

[ARM4R](https://github.com/Dantong88/arm4r.git)生成`4D Representations` ，在T个离散帧表达三维坐标。

[Magma](https://github.com/microsoft/Magma.git)同样是生成`motion seq` ，也可以可视化到图片和视频。这个项目提供了一个VLA相关模型的基础平台，可以用于其他的模型学习。 ⭐

#### Optical Flow

表现的是更密集的表达，公式化为一个动作场(motion field $\mathbf{I} \in\mathbb{R}^{H\times W \times 2}$)。这个场描述的是两帧之间的动作，捕捉整个场景的整体动态性。可以对复杂的、多目标的交互进行显式建模。

##### Key Papers

[AVDC](https://github.com/flow-diffusion/AVDC.git)通过人类或者机器人演示视频和预训练模型训练的`diffusion model` 生成`optical flow` 。

但是`AVDC` ”*is computationally expensive and prone to hallucinations*”，[ATM](https://xingyu-lin.github.io/atm)通过预测轨迹上的任何点优化掉了这个问题。

[Im2Flow2Act](https://im-flow-act.github.io/)不需要任何真实世界机器人数据，从人类演示视频中学习。相比于`ATM` ，这里只关注目标物体的`flow` 。

[FLIP](https://nus-lins-lab.github.io/flipweb/)配合用视频构建的`world model` ，“*It performs model-based planning and predicts action conditioned on both flow and video plan*“
        
###  Goal state
    
预测的未来观测值（比如说图像、点云或者视频片段），可以可视化地表达预测动作序列的预计结果
    
### **Latent representation**
    
有意与训练的一个`latent vector seq` ，其中编码了一些动作相关的信息
    
### **Raw action**
    
控制指令
    
### **Reasoning**
    
自然语言表达（显式的描述做决策的过程来引导特定的动作）