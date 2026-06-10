
# See in the Dark

低照文本分割实验仓。目标不是完整复刻整篇论文，而是把论文里对当前实验最有用的模块拆出来，用一个能在 RTX 4060 Laptop 上跑起来的轻量训练框架验证。

## 项目定位

- 任务：输入低照图像，输出文本区域二值分割结果。
- 形式：PyTorch 模块化训练结构 + 多组 YAML 配置 + PowerShell 启动脚本。
- 风格：以“可快速做消融实验”为主，不追求大而全工程。
- 当前仓库不再提交数据、权重、训练日志和历史结果，这些内容均在本地生成。

## 论文知识点梳理

### 1. 任务本质

低照场景下的文字检测/分割比普通场景更难，主要难点包括：

- 光照弱，前景和背景对比度低。
- 文字边界模糊，细长结构容易断裂。
- 噪声、炫光、阴影会让模型把背景误当成文字。
- 弯曲文本、多方向文本对普通卷积特征不够友好。

这个仓库把问题简化成“像素级文本区域分割”，先解决“看见文字区域”，再谈更完整的检测表示。

### 2. 基线思路

仓库里的基线是一个很小的编码器-解码器分割网络：

- 编码端提取低照图像特征。
- 解码端恢复空间分辨率。
- 主输出头预测文本 mask。
- 损失函数使用 `BCEWithLogitsLoss`。

这部分对应“先得到一个稳定、可训练、可做消融的文本分割基线”。

### 3. SCM：辅助监督与语义一致性

仓库中的 `SCM` 不是完整论文工业级实现，而是一个轻量版辅助分支：

- 在编码特征上接一个辅助头 `aux_logits`。
- `Lsr`：辅助分支直接监督到真实 mask。
- `Lss`：辅助分支输出与主分支概率图做一致性约束。

它背后的论文思想是：

- 主分支只看最终输出不够，训练阶段增加辅助约束可以让特征更稳定。
- 语义一致性让不同分支不要学成彼此冲突的表达。
- 对低照任务来说，这能降低“看不清时特征发散”的风险。

### 4. DSF：方向敏感特征建模

仓库中的 `DSFBlock` 用两条支路做近似实现：

- 一条普通 `3x3` 卷积，保留常规局部纹理。
- 一条 `(1x5) + (5x1)` 的“蛇形/方向性”卷积，增强长条文本结构感知。
- 再用 softmax gate 对两种特征做动态融合。

它对应的论文知识点是：

- 细长文字、弯曲文字不总是适合标准方形卷积核。
- 方向敏感的卷积或可变形/蛇形建模更容易覆盖文字走向。
- 融合时不应固定加权，而应让网络自己决定更依赖哪类特征。

### 5. TSR：文本结构约束与几何塑形

仓库里 `TSR` 分成两部分：

- 训练阶段：额外预测文字中心区域 `center_logits`。
- 可视化阶段：对二值预测做最小外接旋转矩形塑形。

对应的论文思想是：

- 文字不仅是“有没有”，还带有结构中心、长度、走向等几何属性。
- 中心区域监督可以减少文字区域粘连。
- 旋转矩形塑形属于后处理思想，帮助输出更接近规则文本块。

### 6. 损失设计

仓库当前真正落地的损失只有三类：

- 主分割损失：`Lseg`
- SCM 辅助损失：`Lsr + Lss`
- TSR 中心监督损失：`w_tsr_center * center_loss`

需要注意：

- README 中早期提到的更完整组合损失和几何回归项，在当前代码里并没有完整实现。
- 当前仓库更准确的描述是“论文启发式轻量复现”，而不是“论文全量复现”。

## 从头到尾怎么跑

### 1. 安装

```powershell
cd "d:\typer\cursor project\see in the dark"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 准备数据

训练代码默认读取 Paddle 风格 split 文件：

```text
data/raw/ctw1500/paddle_format/ctw1500/imgs/
  ├─ test/
  ├─ training/
  ├─ test.txt
  └─ training.txt
```

其中 `test.txt` / `training.txt` 的每一行格式为：

```text
test/1001.jpg    [{"transcription": 0, "points": [[x1, y1], ...]}]
```

仓库保留了 `scripts/prepare_ctw1500_test.ps1`，它适合下载原始 CTW1500 test 资源；如果要直接训练，仍建议你把数据整理成上面的 Paddle-format 结构。

### 3. 选择配置

- `configs/laptop_4060_quant_friendly.yaml`：最小稳定起步配置。
- `configs/laptop_4060_realdata_ctw1500_smoke.yaml`：真实数据 smoke test。
- `configs/laptop_4060_realdata_ctw1500_1000step.yaml`：真实数据 1000 step。
- `configs/laptop_4060_realdata_ctw1500_high_intensity.yaml`：更激进的训练预算。
- `configs/ablation_*.yaml`：模块消融。
- `configs/grid_*.yaml`：SCM 参数扫描。

### 4. 启动训练

```powershell
.\scripts\start_laptop_train.ps1
.\scripts\start_laptop_train_realdata_smoke.ps1
.\scripts\start_laptop_train_realdata_1000step.ps1
```

### 5. 查看输出

训练结束后会在本地生成：

- `runs/experiments/<experiment_name>/starter_last.pt`
- `runs/experiments/<experiment_name>/train_loss.csv`
- `runs/experiments/<experiment_name>/train_loss_curve.png`
- `runs/experiments/<experiment_name>/prediction_sample.png`

这些文件默认被 `.gitignore` 忽略，不再进入仓库。

## 当前代码实现边界

- 已实现：轻量分割基线、SCM/DSF/TSR 开关、基础可视化、实验配置驱动。
- 已实现：QAT 入口，但默认关闭。
- 未实现：标准检测评估指标、完整论文几何回归、多数据集统一 benchmark。
- 未实现：严格意义上的论文全量复现与正式对标。

## 精简后的目录

```text
see in the dark/
├── configs/
├── scripts/
├── src/
│   ├── see_in_the_dark/
│   │   ├── datasets.py
│   │   ├── models.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   ├── utils.py
│   │   └── main.py
│   └── train_laptop_starter.py
├── .gitignore
├── README.md
├── requirements.txt
└── see text in the dark.pdf
```

其中：

- `datasets.py`：数据集解析与 DataLoader 构建
- `models.py`：TinySegNet、DSF、QAT 相关模型逻辑
- `train.py`：训练循环与损失计算
- `eval.py`：可视化输出与训练产物保存
- `main.py`：读取配置并组装完整训练流程
- `train_laptop_starter.py`：兼容旧脚本的薄入口

## 建议的下一步

- 先补数据准备到 Paddle-format 的自动转换。
- 再补 `IoU / F1 / Hmean` 评估。
- 最后再决定是否继续扩成完整论文版检测框架。
