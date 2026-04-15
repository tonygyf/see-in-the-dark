# CTW1500 数据准备（test 先行）

## 1. 目标
- 先把 `CTW1500` 的 `test` 子集拉到本地，作为“真实数据接入”的第一里程碑。
- 先不改训练主代码，优先把数据目录和下载流程稳定下来。

## 2. 一键命令
在 PowerShell 执行：

```powershell
cd "d:\typer\cursor project\see in the dark"
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_ctw1500_test.ps1
```

可选参数：

```powershell
# 强制重下并覆盖解压目录
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_ctw1500_test.ps1 -Force

# 跳过下载，只用本地 zip 解压（需先把 zip 放到 data/raw/ctw1500/downloads）
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_ctw1500_test.ps1 -SkipDownload
```

## 3. 结果目录
脚本执行后目录应包含：

```text
see in the dark/
  ├─ data/
  │  ├─ raw/
  │  │  └─ ctw1500/
  │  │     ├─ downloads/
  │  │     │  ├─ ctw1500_test_images.zip
  │  │     │  └─ ctw1500_test_labels.zip
  │  │     ├─ imgs/
  │  │     │  └─ test/
  │  │     └─ annotations/
  │  │        └─ test/
  │  └─ processed/
  │     └─ ctw1500/
  └─ docs/
```

## 4. 自检清单
- 终端输出包含：`[Done] CTW1500 test split prepared.`
- `images/test` 和 `annotations/test` 的文件计数均大于 0。
- `data/raw/ctw1500/downloads` 下有两个 zip，后续可离线重复解压。

## 5. 下一步接训练（今天先不改）
- 第一步：在 `src/` 下新增 `Dataset/DataLoader`，先只读取 `ctw1500 test` 做可视化加载检查。
- 第二步：把 `train_laptop_starter.py` 的随机输入替换为真实 batch（先跑 10 个 step 验证显存和 loss）。
- 第三步：记录第一版“真实数据训练日志 + 显存峰值 + 可视化样例”。
