# subremover — 手机端本地去字幕/去水印

目标：完全在手机本地（不走服务器）去掉视频里的字幕、固定水印、动态水印，效果尽量接近云端小程序，做到手机能做到的极限。

## 目录
- `app/`：安卓 App（com.chealert.subremover 0.2.0）。字幕带上光流搬走真实背景；还被挡住的锚点帧用 STTN 原分辨率切块补（`assets/sttn.onnx`，没有则退回 LaMa）。模型和 so 未提交。
- `sttn/`：STTN（researchmm/STTN，MIT）。`pipe.py` 是早期整段缩小的研究流程；`train/` 是手机这条：可加载官方权重、导出给 App 的 ONNX、以及要 GPU 才能跑的字幕微调 / 蒸馏。
- `proto/`：早期原型。

## 现状（样例 a.mp4，28.5 s，1080×1920）
- 字幕、固定水印、动态水印都能稳定识别并去掉，抽查 150 帧无残字。
- 不足：纹理复杂处（玻璃、海报、碗、包装袋）补出来偏糊、有格子感；明显不如小程序。
- 速度：STTN 每次推理（10 帧 × 1 区域）电脑 CPU 约 9 s，全片约 55 min；手机 CPU 估计 70–120 min。0.2.0 手机实测 5 分 28 秒。

## STTN 本身的问题
1. 固定 432×240 低分辨率工作，放大回原图必糊。
2. 固定方块切分 + 注意力，方块间信息交流少 → 格子状涂抹（FuseFormer 论文指出）。
3. 没有光流，只靠注意力找参考内容，运动时易糊（E2FGVI 加光流后明显改善）。
4. 训练用随机/物体掩码，没见过字幕这种细长白字带阴影。

## 路线

手机上跑 STTN，不重训 ProPainter / E2FGVI / FuseFormer。ProPainter 是 CC-BY-NC，不能进这个产品，也远超中高端手机的内存和算力；用它的输出做蒸馏，学生学到的仍是一份非商用模型。E2FGVI、FuseFormer 同样是 GPU 上的大视频修复网。STTN 是 MIT，计算图固定在 432×240，中高端手机的 GPU（ONNX Runtime NNAPI FP16）吃得下。

两截：

1. **推理，已经接到 App。** 光流底板先搬走前后帧里露出来的真实像素。剩下的洞不把整条字幕缩成一张 432×240，而是按原分辨率切成重叠的 432×240 小块，羽化拼回去（`SttnInpainter`，每次 8 帧：附近 6 帧 + 2 帧参考）。没有 `app/assets/sttn.onnx` 时行为和原来一样，锚点用 LaMa。导出：

   ```
   python sttn/train/export_phone.py /path/sttn.pth --t 8 --fp16 --out app/assets/sttn.onnx
   ```

   官方权重、原分辨率切块，和旧的「整段缩小再放大」比（干净画面上叠中文字幕，洞内）：PSNR 23.25 → 23.99 dB，细节能量比（相对原图，1 是一样锐）0.48 → 1.10。糊主要来自缩小，这一截去掉了。4 核 CPU 上一次 8 帧推理约 6.3 秒；一条字幕通常 2–3 块，所以官方权重在手机 CPU 上仍然偏慢，中高端机要靠 NNAPI FP16。

2. **权重，要一块 GPU。** 官方权重没见过字幕，格子和笔画残影还在。`sttn/train/train.py --mode finetune` 在干净视频上叠随机中文字幕（白字、描边、阴影、半透明）微调；`--mode distill --arch s` 再蒸成通道 128、4 层的学生网，大约是官方 1/7 的计算量（34 GFLOP/帧对 238）。图里的注意力改成了最高 4 维张量，NNAPI 可以整段接，不必把 7 维 reshape 丢回 CPU。这台机器没有 CUDA，微调没有在这里跑。

## 新网络 FGFI-Net（分支 `feature/flowformer-inpaint`）

STTN 的上限在结构上，所以重新设计了一个网络：分辨率无关，用软切分代替硬方块，网络内部带光流引导，专门针对字幕和水印。设计、算力、许可和风险都写在 [`docs/new-net-design.md`](docs/new-net-design.md)，代码在 `sttn/newnet/`。这台机器只有 CPU，没有训练过；训练要在租的显卡上跑。

### 在租的显卡上训练

**1. 环境**（CUDA 12.x，Python 3.10–3.12）

```bash
git clone -b feature/flowformer-inpaint https://github.com/winziyuan-design/subremover.git && cd subremover   # 私有仓库：先 gh auth login 或用 token
python -m venv venv && . venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按驱动选 cu 版本
pip install -r sttn/newnet/requirements.txt
sudo apt-get install -y fonts-noto-cjk            # 合成字幕用的中文字体（或用 --fonts 指定）
python sttn/newnet/selftest.py                    # 1 分钟内跑完，最后打印 selftest passed
```

**2. 数据**：只用许可干净的源视频，见设计文档第 5 节。首选 REDS（CC BY 4.0）、Blender 开放电影（CC BY）和自己的无字幕原始素材。训练集和验证集必须来自不同的视频。

```bash
python sttn/train/synth.py prep frames/train  videos/train/*.mp4 --max-short 1080
python sttn/train/synth.py prep frames/val    videos/val/*.mp4   --max-short 1080
python sttn/newnet/data.py frames/train --out preview.jpg     # 先看一眼合成样本是否正常
```

REDS 下载下来是帧目录（`train_sharp/000/*.png`），直接把每段目录放进 `frames/train/` 就行，data.py 认 jpg 和 png。

**3. 训练**：单卡把 `torchrun --nproc_per_node 4` 换成 `python`。`--bs` 是每张卡的批大小，24 GB 显存用 4，40–80 GB 可以用 8。

```bash
export OMP_NUM_THREADS=2
torchrun --nproc_per_node 4 sttn/newnet/train.py --stage 1 --data frames/train --val-data frames/val \
    --out runs/fgfi --bs 4 --workers 10 --bf16
torchrun --nproc_per_node 4 sttn/newnet/train.py --stage 2 --init runs/fgfi/G_latest.pth --data frames/train \
    --val-data frames/val --out runs/fgfi --bs 4 --workers 10 --bf16
torchrun --nproc_per_node 4 sttn/newnet/train.py --stage 3 --init runs/fgfi/G_latest.pth --data frames/train \
    --val-data frames/val --out runs/fgfi --bs 2 --workers 10 --bf16
```

- 中断后用同一条命令重跑，会从 `runs/fgfi/state_stageN.pth` 接着训。
- 日志里每 50 步打印各项损失和 s/it；每 2000 步打印一次 `VAL ... PSNR(hole)`，同时写入 `runs/fgfi/val.jsonl`。
- 数据加载每个样本要在 CPU 上算约 21 次 DIS 光流，所以每张卡要配 8–12 个 CPU 核。如果 GPU 利用率低于 70%，就加 `--workers`，或者换 CPU 核多的机器。

**4. 预计 GPU 时长（估算，没有实测）。** 按每步 FLOP 算：base 配置、192×432、T=8 时，每个 clip 前向加反向约 3.2 TFLOP，其中 G 约 0.9，D 约 1.3，VGG 约 1.0。

| 阶段 | 迭代 | 单卡 A100 每步（bs 4） | A100 时长 |
|---|---|---|---|
| 1（无 GAN/VGG） | 40k | 约 0.1–0.15 s | 1–2 h |
| 2 | 120k | 约 0.25–0.4 s | 8–13 h |
| 3（240×864） | 40k | 约 0.6–1.0 s | 7–11 h |
| 合计 | | | **约 16–26 A100 小时；4090 大约是它的 1.5–2.5 倍；H100 约 0.6 倍** |

多卡基本线性加速，4 卡 A100 约 5–7 小时。上表是第一版能用的模型的量级。如果验证 PSNR 到第 2 阶段末还在涨，可以把 stage 2 加到 200k。先跑 stage 1 的前 2000 步，看实测 s/it，再按比例推算总时长。

**5. 评估。** 在 GPU 机器上跑，或者把权重拷回来在 CPU 上跑：

```bash
python sttn/newnet/eval.py synth --frames frames/val --fgfi runs/fgfi/G_best_s3.pth \
    --sttn sttn.pth --teacher models/teacher_G_latest.pth --lpips --watermark --out eval_synth
python sttn/newnet/eval.py real --video sample/a.mp4 --ref sample/b.mp4 --masks sttn/newnet/realmasks \
    --fgfi runs/fgfi/G_best_s3.pth --sttn sttn.pth --teacher models/teacher_G_latest.pth --out eval_real
```

`sttn.pth` 是 STTN 官方权重，a.mp4 和 b.mp4 是样片，都不在仓库里，需要另外拷过去。

**6. 需要传回来的文件**（总共不到 200 MB）：
- `runs/fgfi/G_best_s3.pth` 和 `runs/fgfi/G_latest.pth`：里面有 EMA 权重，导出时默认用 EMA 权重；
- `runs/fgfi/val.jsonl`；
- `runs/fgfi/args_stage*.json`；
- 训练日志；
- `eval_synth/` 和 `eval_real/` 两个目录。

`state_stage*.pth` 带优化器状态，很大，只有要在别处续训时才需要。

**7. 导出给手机**（App 暂时还没接）：

```bash
python sttn/newnet/export_onnx.py runs/fgfi/G_best_s3.pth --h 204 --w 1080 --t 8 --fp16 --out fgfi_204x1080.onnx
```

脚本会打印算子表、不在 NNAPI 支持范围内的算子（只有 GridSample）、最大张量维数（4），以及 ORT 和 torch 的输出一致性对比。
