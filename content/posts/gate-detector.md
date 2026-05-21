---
title: "基于强化学习的穿越机-GateDetector篇"
date: 2024-10-19T16:13:11+08:00
tags:
  - Deep Learning
  - pyTorch
math: true
---
## 简介

该篇blog记录了项目《自主导航的穿越无人机研究》的开发全过程，如有想法，欢迎发送邮件或者在github下给我评论。

## 2024.10.4

### MIoU 语义分割指标

1. IOU，交并比：真实标签和预测值的交和并的比值![image-20241004223149642](/images/image-20241004223149642.png)公式：    $$ IoU = TP / (TP+FN+FP) $$

2. MIoU是该数据集中的每一个类的交并比的平均
   $P_{ij}$表示将$i$类别预测为$j$类别
   $$
   MIoU=\dfrac{1}{k+1}\sum\limits_{i=0}\limits^{k}\dfrac{P_{ii}}{\sum_{j=0}^{k}p_{ij}+\sum_{j=0}^kp_{ji}-p_{ii}}
   $$
   等价于
   $$
   MIoU=\frac{1}{k+1}\sum\limits_{i=0}^k\dfrac{TP}{FN+FP+TP}
   $$

3. 混淆矩阵
   ![image-20241004224956545](/images/image-20241004224956545.png)

   ```python
   '''
   产生n×n的分类统计表
   参数a：标签图（转换为一行输入），即真实的标签
   参数b：score层输出的预测图（转换为一行输入），即预测的标签
   参数n:类别数
   '''
   def fast_hist(a, b, n):
       #k为掩膜（去除了255这些点（即标签图中的白色的轮廓），其中的a>=0是为了防止bincount()函数出错）
       k = (a >= 0) & (a < n) 
       return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)
   
   ```

4. 利用混淆矩阵计算iou和miou

   ```py
   def per_class_iu( hist ): #
       # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
       return np.diag( hist ) / ( hist.sum( 1 )  + hist.sum( 0 ) - np.diag( hist ))  
   ```

### code 

1. 添加合适的自调整学习率

   ```py
   
   #  定义Adam算法
   optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas,
                              weight_decay=weight_decay)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                        factor=0.8, patience=15, verbose=True)
   ```

## 2024.10.6

### 新UI

尝试训练5000 epochs。是个超长的炼丹......🪄
![image-20241006115719284](/images/image-20241006115719284.png)
但是新UI很好看😍

### 新增验证集和自调整学习率

一般的训练是训练和验证集同时进行，偷懒，之前没有加上，今天左思右想，还是得给加上，虽然既有的test效果很好，但是要做一个严谨的researcher。

```py
    # 加载训练集和验证集
    dataset = MyDataLoader(data_path, num_classes=classesnum)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    
    ......
    
            # 评估模式（不计算梯度）
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, label in val_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)

                pred = net(image)
                loss = criterion(pred, label)
                val_loss += loss.item()    

```

然后用验证集每一个epoch的平均loss 调整自学习率。

```py
scheduler.step(avg_val_loss)
```

最后在UI和TensorBoard上加入验证集部分
![image-20241006122130673](/images/image-20241006122130673.png)
更加炫酷😉😉

### 改变Loss计算方式
目前Loss计算方式 交叉熵计算

```py
nn.CrossEntropyLoss()
```

通过学习*《pyTorch深度学习实践》*，获知**dice loss（骰子损失）**是图像分割惯用的一种计算损失的方式，因此在网上学习。
感谢博客（[图像语义分割训练经验总结--图像语义分割 - Rser_ljw - 博客园 (cnblogs.com)](https://www.cnblogs.com/ljwgis/p/12313047.html)）
其中对图像分割二分类、多分类问题分别概括了。
二分类问题：
![image-20241006122854092](/images/image-20241006122854092.png)
（ps：因为我的GateDetector是多分类问题，识别门的上、下、左、右，因此跳过二分类问题，这里留下以便后面重温的时候温故而知新)
多分类问题：
![image-20241006123224797](/images/image-20241006123224797.png)
这里讲了一些关于多分类问题要注意的点。其中提到了损失函数CELoss的输入输出问题，但是我已经将label以ONE-HOT编码输出了，事实证明不影响Loss的计算。
再者就是博客（[分割网络损失函数总结！交叉熵，Focal loss，Dice，iou，TverskyLoss！_tversky loss-CSDN博客](https://blog.csdn.net/jijiarenxiaoyudi/article/details/128360405))讲解了Dice Loss的计算应该和CE Loss一起搭配使用
![image-20241006123611891](/images/image-20241006123611891.png)

原作者博客：[常用损失函数（二）：Dice Loss-CSDN博客](https://blog.csdn.net/Mike_honor/article/details/125871091)
骰子损失函数：

```py
def dice_loss(pred, target):
    smooth = 1e-6
    pred = F.softmax(pred, dim=1)
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    intersection = (pred_flat * target_flat).sum()
    dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_score
```

加上结合损失函数：（结合了CE Loss和Dice Loss）

```py
def combined_loss(pred, target, alpha=0.5):
    target_indexed = torch.argmax(target, dim=1)
    ce_loss = nn.CrossEntropyLoss()(pred, target_indexed)
    d_loss = dice_loss(pred, target)
    return alpha * ce_loss + (1 - alpha) * d_loss
```

### 重新炼丹🪄🪄🪄...

![image-20241007154047512](/images/image-20241007154047512.png)
炼丹完成😍😍😍😍

### 模型验证

![img0_20240924_161725](/images/img0_20240924_161725.png)![img0_20240924_161725_res](/images/img0_20240924_161725_res.png)![img0_20240924_161728](/images/img0_20240924_161728.png)![img0_20240924_161728_res](/images/img0_20240924_161728_res.png)![img0_20240924_161745](/images/img0_20240924_161745.png)![img0_20240924_161745_res](/images/img0_20240924_161745_res.png)

## 2024.10.9

### 思考问题

1. 该如何获取图像反馈回来的坐标？
   plan A：实时获取，通过图像获取像素坐标。使用图像处理算法实时跟踪
   plan B：第一时间获取，后续进行更新，运用kalman filter滤波后续数据

### 问题解决

返读论文*《Champion-level drone racing using deep reinforcement learning》*，其中Methods部分的**VIO drift estimation**![image-20241009114919752](/images/image-20241009114919752.png)划线部分再次强调Gate Detector的作用--输出Gate的四角坐标，通过IPPE（[tobycollins/IPPE](https://github.com/tobycollins/IPPE)）来估计相对位置。

## 2024.10.16

> 休息了很久，重新开始工作

### 目前图像预测结果

目前图像预测结果非常的不稳定
![img0_20240924_161825_res](/images/img0_20240924_161825_res.png)
图中分割区域非常不完整且不连续，需要进一步对图像处理。

### Post-Processing 后处理

#### 图像形态学 - 开运算

尝试用了个20*20大小的全1核

```python
import cv2
kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (20, 20) )
img = cv2.morphologyEx( img, cv2.MORPH_OPEN, kernel )
```

处理效果如下
![image-20241016151111830](/images/image-20241016151111830.png)

#### 区域连通域分析

查阅了多个资料，最终思路：在label图上将不同的label单独分离出来，随后进行轮廓检测，然后计算面积去除小面积的区域。
![image-20241016164624026](/images/image-20241016164624026.png)
分离出来后的每个label的二值图，随后先做膨胀和高斯平滑，填充部分图内的空缺，进行最大轮廓识别

```py
def getMaxContour(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, kernel)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    img_contours = np.zeros([480, 640], np.uint8)
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    contours_max = np.argmax(np.array(area))
    cv2.drawContours(img_contours, contours, contours_max, 255, cv2.FILLED)
    return img_contours
```

![image-20241016180018787](/images/image-20241016180018787.png)
最后提取出来的图像如上图，最后合并回原图。
![image-20241016182556331](/images/image-20241016182556331.png)
最终结果如上

## 2024.10.17

### 一些思考...

> 思考1：我能不能计算两个区域的距离，随后进行图像填充
> 思考2：或许我是否能通过计算像素点的最小距离的中值点来确定我需要的坐标点

### 《图片修复》相关的算法

**PatchMatch** 图像修复算法，感觉这类算法并不能解决现在遇到的问题，因为需要参考图Mask。

提出可能：建立Mask — 在飞行过程中对门直接整体轮廓识别，随后将预测图**区域连通域分析**的结果和整体轮廓识别后的结果进行图像加减得到Mask图，使用PatchMatch做图像修复。

### 关于思考一

思考一的**距离检测**能不能拿来做主成分和副成分的筛选，这样就不用无脑删掉处理除最大面积以外的轮廓块。
