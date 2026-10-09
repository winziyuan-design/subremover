# FGFI-Net：字幕 / 水印专用的视频修复网络（设计文档）

分支 `feature/flowformer-inpaint`，代码在 `sttn/newnet/`。FGFI = **F**low-**G**uided **F**use-transformer **I**npainter。

## 0. 为什么要换结构

STTN（MIT）在我们的样片上到不了小程序的效果，原因在结构本身，换数据也去不掉：

| STTN 的硬伤 | 后果 | FGFI-Net 的做法 |
|---|---|---|
| 网络内部固定 432×240（PATCHES 写死） | 只能切块或缩放；缩放必糊 | 全卷积编解码 + 1/4 分辨率 token，任意 H%12、W%72 的条带直接按原分辨率推理 |
| 硬方块切分 + 注意力 | 格子感、块边接缝 | FuseFormer 式软切分 / 软合成：7×7 重叠 patch，步长 3；FFN 里再做一次合成和切分（F3N） |
| 网络内没有光流 | 一动就糊；从别的帧找内容全靠注意力去猜 | 网络内补全光流 → 光流引导的双向特征传播（warp + 一致性门控）→ 注意力的 Q/K 里带光流嵌入 |
| 没见过字幕 | 笔画残影、阴影残边 | 只用合成字幕 / 水印数据训练（复用 `synth.py`） |

不走蒸馏：蒸馏学生的上限就是老师。

## 1. 输入输出（与 App 的接口）

一次推理处理一个 **T=8 帧窗口里的一条条带**（band，例如 1080×204 的字幕带；竖排水印旋转 90° 后也当横条处理）。

- `frames` [T,3,H,W]：RGB，0..1。
- `masks` [T,2,H,W]：第 0 通道是还不知道内容的洞；第 1 通道是 App 的像素级光流传播已经搬进真实背景的洞（prefilled）。网络把两者都当成洞来处理，但 prefilled 的像素可以直接参考，网络只负责修接缝和误差。没有 prefill 时第 1 通道传全 0。
- `flow_fw` / `flow_bw` [T-1,2,H/2,W/2]：相邻帧的前向（t→t+1）和反向（t+1→t）光流，单位是半分辨率像素。就是 App `FlowField` 现在算的东西：在带字幕的原始条带上跑 DIS（PRESET_FAST，0.5 倍尺寸），洞里用归一化卷积填。训练数据也用同一套方法生成（`data.py` 的 `nc_fill`），保证训练和手机上的分布一致。
- 输出 `out` [T,3,H,W]：洞外是原图，洞内是修复结果。
- 形状是静态的，NNAPI 需要静态形状。H 必须是 12 的倍数，W 必须是 72 的倍数，App 通过扩大条带上下文来满足，不在图里 pad。1080 和 720 宽正好都是 72 的倍数。常用的尺寸各导出一个 onnx，比如 204×1080、144×1080、240×1080。

## 2. 网络结构（`model.py`，配置 `base`）

```
x(5ch: RGB+2 mask) ─ e1 conv3x3 (24) ──────────────────────────────── skip1
                     e2 s2 conv (48) ─────────────────────────── skip2      (1/2)
                     e4 s2 conv + res (96)                                   (1/4)
flow_fw/bw (1/2) ─ avgpool→1/4 ─ FlowCompletion(dilated U-Net, residual only inside dilated hole)
                                 └─ fw-bw consistency → validity v
特征传播（1/4）：反向递推 t=T-2..0: h_t = f_t + Conv[f_t, warp(h_{t+1}, F_t→t+1)·v, v, m, F]
                 正向递推 t=1..T-1:   g_t = h_t + Conv[h_t, warp(g_{t-1}, F_t→t-1)·v, v, m, F]
                 out_t = f_t + 1x1[h_t, g_t]
软切分：Conv2d(96→192, k7, s3, p3)  ≡ unfold(7×7, stride 3) + Linear   → token 图 [BT,192,H/12,W/12]
光流嵌入：Conv2d(7→192, k7, s3)（前/后向补全光流、两个一致性图、洞）
8 × Block:
    x += DWConv3x3(x)                       条件位置编码，分辨率无关
    x += WindowAttn(Norm(x), flow_emb)      窗口 = 全部 T 帧 × 条带全高 × 6 列 token；奇数块平移 3 列
                                            Q,K = 1x1([x, flow_emb])，V = 1x1(x)
    x += F3N(Norm(x))                       ConvT(192→32,k7,s3)/重叠数 → LeakyReLU → Conv(32→192,k7,s3)
软合成：ConvTranspose2d(192→96, k7, s3)/重叠数 ≡ Linear + fold，再与传播特征拼接
解码：1/4 → 最近邻上采样 + conv → [skip2] → 上采样 + conv → [skip1] → conv → tanh
```

要点：

- **分辨率无关。** 除了窗口宽（6 列 token = 72 输入像素）以外没有固定尺寸。训练用 192×432 / 240×864 的裁剪，推理直接跑 1080 宽的整条带，不切块，也没有块缝。条带高度通常 15–20 行 token，窗口覆盖全高，所以竖直方向（字幕洞最窄的方向）每个窗口都能看到洞的上下两侧。
- **软切分和 F3N** 按 FuseFormer 论文（Liu et al., ICCV 2021）的定义自己实现：unfold+Linear 在数学上等价于 Conv2d(k7,s3)，Linear+fold 等价于 ConvTranspose2d(k7,s3)，再除以每个位置被覆盖的次数（常量图）。这样导出后只有标准的 Conv / ConvTranspose，没有 Im2Col。FuseFormer 仓库**没有 LICENSE**，等于保留所有权利，所以一行代码都没有抄。
- **光流引导** 参考 FGT（Zhang et al., ECCV 2022，MIT）和 E2FGVI（Li et al., CVPR 2022，CC BY-NC，只看论文）的思路，代码自己写：先补全光流；用补全后的光流做双向 warp 传播，用前后向一致性图作门控（E2FGVI 用的是可变形卷积，手机后端导不出，这里改成 warp+卷积）；注意力里的光流嵌入相当于 FGT 的 flow-guided attention，窗口限定在 T 帧的局部时间范围内。
- **算子约束。** 所有张量不超过 4 维：窗口划分用「先把 ws 和 d 合并成一维再 permute」的写法。没有 LayerNormalization（ChannelNorm 用逐元素算子写）、没有 Gelu/Erf、没有 Im2Col，也没有可变形卷积。训练时注意力用 SDPA，导出时换成 MatMul+Softmax。导出后经 ORT basic 优化，图里只剩这些算子：Conv、ConvTranspose、MatMul、Softmax、Slice、Concat、Reshape、Transpose、Add/Sub/Mul/Div、LeakyRelu、ReduceMean、Sqrt、Exp、MaxPool、AveragePool、Resize、Clip、Tanh、Neg、**GridSample**。只有 GridSample（光流 warp，共 16 个）不在 NNAPI 的支持范围内，会回落到 CPU EP。它只在 1/4 分辨率上做，计算量很小，代价是几次 NNAPI↔CPU 分区切换（见风险 3）。

### 规模（实测，`export_onnx.py` / FlopCounter）

| 配置 | 参数 | 8×(204×1080) 一个窗口 | 每帧 | 盒子 CPU ORT 4 线程 |
|---|---|---|---|---|
| small | 4.2 M | 355 GFLOP | 22 GMAC | 1.22 s |
| **base** | **10.5 M** | **810 GFLOP** | **51 GMAC** | **2.1–2.3 s** |
| STTN 原版（参照） | 16.6 M | 8×240×432 一块 2063 GFLOP；1080 宽的条带要 3 块，约 6.2 TFLOP | 129 GMAC/块帧 | 每块 4.41 s，3 块 13.2 s |

同一条 1080 宽的字幕带，base 的计算量约为 STTN 切块的 1/7.6，盒子 CPU 上实测快约 6 倍。在质量上限更高的前提下也更快。

### 手机耗时估计（骁龙 870，未实测，诚实区间）

- **CPU**（XNNPACK/CPU EP，4 个大核 fp32）：按盒子 CPU 和 A77 的单核比例粗算，每窗口 **6–12 s**。
- **NNAPI fp16（Adreno 650）**：0.81 TFLOP，有效吞吐按 0.3–0.5 TFLOPS 算是 1.6–2.7 s；加上 16 次 GridSample 回落 CPU 的分区开销，每窗口 **约 2–4 s**。fp16 导出时 GridSample 保持 fp32（`op_block_list`）。
- 一个窗口 8 帧，30 fps 的视频，每秒约 4 个窗口（窗口之间有重叠时更多）。只在有字幕的片段推理，并且 App 的像素传播能搬走的帧不进网络。如果 base 还是太慢，就用 `small`（参数量和计算量约为 base 的 0.44 倍），或者把窗口改成 T=6。
- 比较实在的验证方法：导出 204×1080 的 fp16 onnx，在 vivo 上用 ORT 的 `onnxruntime_perf_test` 或 App 内的计时实测。这一步留到下一阶段接 App 时做。

## 3. 光流

- **手机推理**：OpenCV DIS（Apache-2.0），App 已有，零新增依赖。网络内的 FlowCompletion 负责修补洞里的光流。
- **训练**：输入光流和手机完全一样，是在带字幕的条带上跑 DIS-fast 再做归一化卷积；它在数据加载的 CPU worker 里计算，每个样本约 30–50 ms。监督光流用干净帧上的 DIS-medium，供光流补全损失和时间一致性损失使用。可选升级是用 RAFT（princeton-vl，BSD-3）在 GPU 上算干净帧光流，监督会更准，但它的预训练权重是在 FlyingChairs/Things/Sintel/KITTI 上训的，这些数据集许可不一，好在只用于训练监督，不进产品。v1 先用 DIS，零许可风险。
- FlowCompletion：输入为前后向光流 + 两帧洞，结构是 1/4→1/8 的空洞残差 U-Net，只在膨胀后的洞内输出残差（输出层零初始化，起点就等于 App 现在的归一化卷积补全）。

## 4. 损失（`losses.py`）

| 项 | 权重（stage 2/3） | 说明 |
|---|---|---|
| hole L1 | 1.0 | 洞内（unknown ∪ prefilled） |
| valid L1 | 0.5 | 洞外，稳定重建 |
| 高频 Laplacian L1 | 1.0 | 直接惩罚糊（沿用 v1 train.py） |
| VGG16 感知 + style | 0.05 / 40 | 沿用 v1 `VGGLoss`（torchvision ImageNet 权重，只用于训练） |
| T-PatchGAN hinge | 0.01 | 判别器沿用 `sttn/train/net.py`（STTN, MIT），real/fake 拼在一起前向，兼容 DDP |
| 时间 warp 一致性 | 0.5 | comp_t 与 warp(comp_{t+1}, 干净光流)，限一致性通过且在洞附近 |
| 光流补全 L1 | 0.25 | 补全光流 vs 干净帧光流，洞内 ×5 |

Stage 1 只开 L1 / 高频 / 光流（先学会重建，GAN 不稳定）。

## 5. 训练数据（`data.py`，复用 `synth.py`）

- 合成：复用 `synth.py` 的字幕样式（白字、黄字、彩色字、描边、阴影、半透明底条、半透明）、渲染器和手机 StrokeMasker 风格的膨胀掩码。在此基础上新增：
  - 1–2 行字幕，窗口中途换行；
  - 静态水印和**移动水印**（线性漂移，模拟「花木林」），半透明，可带描边或阴影，文字是汉字、@账号、「xx号:abc123」或拉丁字母；
  - 掩码膨胀系数随机（0.7–1.6），5% 是检测框式的矩形洞；
  - 输入做 JPEG 压缩（q55–95）；
  - **prefill 模拟**：洞内一部分随机区域放入偏移 ±1.5 px、亮度 ±5%、可能轻微模糊的真实背景，标成 prefilled，模拟 App 像素传播后的残差；
  - 时间步长 1 或 2，水平翻转，尺度 0.75–1.25；
  - 裁剪尺寸任意（H%12、W%72）。
- 课程学习：`level` 从 0.2 升到 1.0，控制水印、移动水印、双行字幕、prefill 和 JPEG 的出现概率。由主进程更新共享内存，worker 实时读取。
- **干净视频来源（许可）**：

| 来源 | 许可 | 结论 |
|---|---|---|
| **REDS**（NTIRE 2019，720p，270 段动态场景） | CC BY 4.0 | ✅ 首选，注明出处即可 |
| **Blender 开放电影**（Big Buck Bunny、Sintel、Tears of Steel、Spring、Cosmos Laundromat） | CC BY | ✅ 可用，但偏 CG，占比控制在 20% 以内 |
| **自己拍或已获授权的无字幕原始素材**（食品、产品、手持、室内暖光，最接近样片） | 自有 | ✅ 最推荐，几十段就很有价值 |
| DAVIS 2017 | 代码 BSD，标注 CC BY 4.0，视频本身许可不清（HF 镜像标为 CC BY-NC） | ⚠️ 只建议做验证，不进训练 |
| YouTube-VOS | 仅限非商业研究，视频版权属于上传者 | ❌ 不进商用训练 |
| Pexels / Pixabay | Pexels 条款明确禁止把内容用于机器学习数据集或训练（已查证）；Pixabay 未逐条核实 | ❌ 不用 |
| Vimeo-90K、Inter4K 等 | 研究用途或 NC | ❌ |

  注意：现有的 `models/teacher_G_latest.pth` 是用什么数据微调的，仓库里没有记录。如果用了研究用途的数据，它也只能作为评估基线，不能进产品。
- 验证集必须来自**不同的源视频**（`--val-data`），否则 PSNR 虚高。
- 预处理：`python sttn/train/synth.py prep frames/train videos/*.mp4 --max-short 1080`（每段视频解码成一个 JPEG 帧目录）。

## 6. 训练流程（`train.py`）

| 阶段 | 裁剪 | 迭代 | lr | 损失 | 难度 level |
|---|---|---|---|---|---|
| 1 预热 | 192×432 | 40k | 2e-4 | L1 + hf + flow | 0.2→0.6 |
| 2 主训练 | 192×432 | 120k | 1e-4 | 全部 | 0.6→1.0 |
| 3 大裁剪 | 240×864 | 40k | 3e-5 | 全部 | 1.0 |

- 优化器：AdamW (0.9, 0.99)，warm-up 后按 cosine 衰减；EMA 0.999（推理和导出用 EMA 权重）；梯度裁剪 1.0；bf16 autocast；DDP（torchrun）。
- 断点续训：每 2000 次迭代写 `state_stageN.pth`，原子替换。重启时用同一条命令即可续上。
- 验证：每 2000 次迭代在 held-out 数据上算洞内 PSNR、SSIM 和细节比，写进 `val.jsonl`，最好的一版存为 `G_best_sN.pth`。

## 7. 评估协议（`eval.py`）

1. **合成 held-out**（`eval.py synth`）：在干净的高清 held-out 片段上，按原分辨率叠加中文字幕（可加 `--watermark` 叠移动水印），每段取 3 个窗口。比较的方法：
   - FGFI-Net；
   - STTN 官方权重的原分辨率切块；
   - 字幕微调过的 teacher 切块。

   三者走同一套流程，指标都只在洞内算：PSNR、SSIM、LPIPS（`--lpips`，需要装 `lpips` 包）、细节能量比（Laplacian 能量相对原图，1.0 表示一样锐，小于 1 偏糊，大于 1 有噪声或振铃）、每窗口耗时。
2. **真实样片**（`eval.py real`）：a.mp4 的四个镜头，即玻璃杯 3.8 s、海报 12 s、碗 18 s、包装袋 28 s，每个镜头取一个 8 帧窗口。掩码来自 `pipe.py`，已经放进 `sttn/newnet/realmasks/`，一共 32 张。每个镜头输出最大的两个区域（字幕带和动态水印）的并排对比图，依次是：输入、FGFI、STTN 切块、teacher 切块、小程序 b.mp4。b.mp4 是 720p，对比时放大到 1080p。此外还输出洞内细节能量相对 b.mp4 的比值。
3. **通过标准**（进 App 的门槛）：
   - 合成集上，洞内 PSNR 比 teacher 切块高 ≥ 1 dB，并且细节比落在 0.8–1.2；
   - 真实四个镜头肉眼看不出格子和残字，细节能量 ≥ 0.8×b.mp4；
   - 手机实测每窗口 ≤ 4 s（NNAPI）。

## 8. 风险

1. **数据量和域差。** 许可干净的源视频少，REDS 只有 720p，和竖屏美食、产品视频的域差不小。最有效的补救是自有的无字幕原始素材。
2. **从零训练。** 没有可以商用的预训练权重，结构也和 STTN 不同，不能继承 teacher。第一版可能需要比估计更多的迭代。
3. **NNAPI 分区。** 16 个 GridSample 会把图切成大约 17 段，在 GPU 和 CPU 之间来回切。如果实测开销大，有两个备选：一是把传播模块拆成单独的 CPU 图，编码器和 transformer 走 NNAPI；二是改走 MNN 的 OpenCL 后端（GridSample 有 GPU 实现，项目里已经有 MNN 导出经验）。
4. **大运动。** 窗口只有 72 px 宽，大位移要靠光流传播先对齐；DIS-fast 在洞附近和大运动时误差大，只能靠 FlowCompletion 和一致性门控兜底。
5. **整窗时序。** 窗口之间没有递归状态，窗口边界可能闪烁。App 侧要重叠 2 帧并做交叉淡化，这一块还没做。
6. **GPU 小时数是估算**，见 README。

## 9. 许可小结

- 我们的代码：自写。
- 引用但没有拷贝代码的工作：STTN（MIT，判别器和 VGGLoss 来自本仓库已有的 MIT 衍生代码）；FGT（MIT，只参考思路）；FuseFormer（无许可证，只按论文实现）；E2FGVI（CC BY-NC，只看论文）。
- 推理依赖：OpenCV DIS（Apache-2.0）、ONNX Runtime（MIT）。
- 只在训练时用到：torchvision VGG16 ImageNet 权重（损失用，不发布）；RAFT 可选（BSD-3）。
- 只在评估时用到：LPIPS（BSD-2）。
