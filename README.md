
# 🌙 See in the Dark

<p align="center">
  <b>Low-Light Text Segmentation (PyTorch)</b><br/>
  在低照环境中“看见文字”的模型复现与实验平台
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c"/>
  <img src="https://img.shields.io/badge/GPU-RTX%204060%20Laptop-76b900"/>
  <img src="https://img.shields.io/badge/Platform-Windows-0078d4"/>
</p>

---

## ✨ Highlights

- ⚡ 轻量复现框架：专为 **Laptop GPU (RTX 4060)** 优化  
- 🧩 模块化设计：TSR / DSF / SCM 可独立开关  
- 📊 自动记录：loss 曲线 + 可视化输出  
- 🔁 快速实验：一键脚本切换配置  

---

## 🧠 What This Project Does

> 给定低光图像 → 输出文本区域分割 Mask

模型输出包括：

- `Pred Prob`：像素级概率图（0~1）
- `Pred Binary`：阈值后的文本区域

---

## 🚀 Quick Start

```powershell
cd "d:\typer\cursor project\see in the dark"

# 创建环境
python -m venv .venv
.\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 准备数据
.\scripts\prepare_ctw1500_test.ps1

# 开始训练（1000 step）
.\scripts\start_laptop_train_realdata_1000step.ps1
````

---

## 📁 Project Structure

```text
see in the dark/
│
├── configs/              # 实验配置（yaml）
├── scripts/              # 一键脚本（PowerShell）
├── src/                  # 核心训练代码
│
├── runs/
│   ├── experiments/      # 当前实验输出
│   └── history/          # 历史记录
│
├── checkpoints/          # 模型权重（.pt）
└── README.md
```

---

## 📊 Current Results

| Experiment   | Final Loss ↓ |
| ------------ | ------------ |
| **TSR Only** | **0.495868** |
| DSF Only     | 0.516334     |
| All Modules  | 0.656898     |
| SCM Only     | 0.759101     |

### 🔍 SCM Sweep

| Config    | Loss         |
| --------- | ------------ |
| scm_w10_5 | **0.580900** |

---

## 🧪 Experiment Philosophy

* 单模块验证 → 联合训练
* 固定预算（step / batch）做公平对比
* 所有实验可复现（script + config）

---

## 📦 Data

* 当前数据集：**CTW1500（Paddle 标注格式）**

### 🔗 扩展数据源

* Text-in-the-Dark
  [https://github.com/chunchet-ng/text-in-the-dark](https://github.com/chunchet-ng/text-in-the-dark)

* LATeD Paper
  [https://arxiv.org/pdf/2404.08965](https://arxiv.org/pdf/2404.08965)

---

## 📈 Outputs

训练过程中自动生成：

* 📉 `train_loss_curve.png`
* 📄 `train_loss.csv`
* 🖼 `prediction_sample.png`

可视化包含：

| 输入 | GT Mask | Pred Prob | Pred Binary |
| -- | ------- | --------- | ----------- |

---

## 🧭 Roadmap

### 🔜 Next Steps

* [ ] 加入评估指标：F1 / IoU / Hmean
* [ ] 固定 `SCM w10_5` 扫联合参数
* [ ] 接入低照数据（dark dataset）
* [ ] 做 cross-dataset 对比

---

## ⚙️ Training Entry

```bash
src/train_laptop_starter.py
```

配置驱动：

```bash
configs/*.yaml
```

---

## 💡 Tips

* 初始训练（<100 step）预测全黑是正常现象
* 推荐训练步数：`≥1000`
* 可调阈值观察模型学习情况（0.3 / 0.5 / 0.7）

---

## 🧑‍💻 Author Notes

这个项目的目标不是“跑通”，而是：

> ⚡ 在有限算力下，把实验做快、做准、做清楚

---

<p align="center">
  🌘 from darkness → to structure → to understanding
</p>
```

---

