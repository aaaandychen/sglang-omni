# SGLang-Omni 入门指南

> 假设你已懂 LLM 推理（prefill/decode、KV cache、scheduler）。
> 目标：跑通服务，然后理解 Omni 最核心的一件事——多 stage 流水线是怎么组织、怎么通信的。

---

Omni 模型相比 LLM 多了三件事：
1. **多模态输入 / 多模态输出**：文本、图像、音频、视频进；文本 / 音频 / 波形出。
2. **多 stage 流水线**：单体 `Scheduler + ModelRunner` 不再够用。典型链路：
   ```
   preprocessing → image/audio encoder → mm_aggregate → thinker (AR) → talker (AR) → code2wav (vocoder)
                                                                    ↘ decode (text)
   ```
3. **跨进程 / 跨 GPU 通信**：stage 间需要传 tensor，引入 **Control Plane（ZMQ 小消息）+ Data Plane（Relay：SHM/NCCL/NixL/Mooncake/CUDA IPC/LOCAL_OBJECT）**。

---

## 1. 先跑起来

```bash
# Docker（推荐）
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all --ipc host --network host --privileged \
    frankleeeee/sglang-omni:dev /bin/zsh

# 容器内
git clone git@github.com:sgl-project/sglang-omni.git && cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate && uv pip install -v -e .
```

纯文本测试：
```bash
sgl-omni serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --host 0.0.0.0 --port 8000

curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

带音频输出（体会多 stage）：
```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 --gpu-talker 1 --gpu-code2wav 1 --port 8000
```

启动日志里你会看到 `preprocessing`、`thinker`、`talker_ar`、`code2wav` 各自启动、各自绑定 GPU。**这就是 Omni 和单体推理最直观的差异：一个请求要经过多个 stage，stage 分布在不同进程/GPU 上。**

---

## 2. 核心问题：为什么需要 Pipeline + Stage？

LLM 推理是单体的：一个 Scheduler + 一个 ModelRunner + 一次 forward 搞定。

但 Qwen3-Omni 这种模型，一次"说话"实际要走：

```
预处理 → 图像/音频编码 → 多模态融合 → Thinker(AR生成文本+hidden states)
                                           ├→ Decode(输出文本)
                                           └→ Talker(AR生成audio codes) → Code2Wav(声码器输出波形)
```

这些步骤计算特征不同、设备需求不同、并行度不同。如果塞进一个进程，调度器要同时管 KV cache、管 streaming codes、管声码器——复杂度爆炸。

所以 Omni 的做法是：**拆成多个 Stage，每个 Stage 只管一件事，Stage 之间通过消息和共享数据通信。**

---

## 3. Stage 是什么

一个 Stage 本质上就是：

```
        ZMQ 控制消息
        ┌─────┐
        │     ↓
 Relay → Stage → Scheduler → ModelRunner（可选）
 数据面   ↑     │
        │     ↓
        ZMQ 控制消息（给下游）
```

**Stage 是做 IO 的外壳**，它不感知模型。它的工作循环是：

1. 从 ZMQ 收控制消息（Submit / DataReady / Stream / Abort）
2. 如果需要拿数据，通过 Relay 读 tensor
3. 把工作塞进 `Scheduler.inbox`
4. 从 `Scheduler.outbox` 取结果
5. 如果要发给下游，通过 Relay 写 tensor，再发 ZMQ 控制消息通知下游来取

代码在 `pipeline/stage/runtime.py`，看懂这个循环就懂了 Stage。

**关键约束：每个 Stage 有且仅有一个 Scheduler。** Stage 不关心 Scheduler 是什么类型，只管往 inbox 塞、从 outbox 取。

---

## 4. Stage 之间怎么通信

这是整个框架最核心的设计。

### 两条通道

```
Stage A                              Stage B
  │                                     │
  │── ZMQ 小消息（控制面）────────────→ │  "你的数据准备好了，去 relay 取"
  │                                     │
  │── Relay 大 tensor（数据面）───────→ │  实际的 tensor buffer
```

**控制面（ZMQ）**：传几十字节的消息，告诉对方"有新请求""数据准备好了""流式来了一帧""abort"。消息类型：`SubmitMessage`、`DataReadyMessage`、`StreamMessage`、`AbortMessage`。

**数据面（Relay）**：传大的 tensor buffer。有多种后端可以选：

| 场景 | Relay 后端 | 怎么选 |
|---|---|---|
| 本地多进程 | `shm`（共享内存） | 默认，最简单 |
| 跨 GPU | `nccl` | GPU RDMA |
| 高性能跨机 | `nixl` / `mooncake` | 生产环境 |

### 一次数据传输的完整过程

```
1. Sender: 把 tensor 写入 relay → relay.put_async(tensor)
2. Sender: 通过 ZMQ 发 DataReadyMessage，里面带 relay metadata（告诉对方去哪读）
3. Receiver: 收到 DataReadyMessage → 根据 metadata 调 relay.get_async() 拿回 tensor
```

**关键点：先发控制消息再完成 put。** 如果反过来，某些 backend（如 NIXL）会因为 receiver 永远不启动读而死锁。

### 同进程不走 Relay

如果两个 Stage 在同一个 OS 进程（通过 `fused_stages` 配置），直接传 Python 对象引用，跳过 relay 和 ZMQ。这叫 `LOCAL_OBJECT` 路径——最快，但要求接收方把对象当只读。

同 GPU 的流式传输走 `CUDA IPC`，也是直接共享显存，不打 relay。

---

## 5. Scheduler：连接 Stage 和实际计算

Stage 只管收发，**真正的计算调度在 Scheduler**。Omni 有三种 Scheduler：

**OmniScheduler**（AR stage 用，如 thinker、talker）：
核心思路——**组合** sglang 上游 Scheduler，不是继承。复用它成熟的 KV cache 管理、prefill/decode 调度、batch 选择。但把 IO 全部替换：不从 ZMQ 收请求（从 Stage inbox）、不往 tokenizer 发结果（往 Stage outbox）。代码在 `scheduling/omni_scheduler.py`。

**SimpleScheduler**（非 AR stage 用，如预处理、编码器）：
没有 KV cache，没有 batching。循环就是 `inbox.get() → 处理函数 → outbox.put()`。

**Code2WavScheduler**（流式声码器用）：
三种事件驱动：新请求初始化状态、stream_chunk 累积解码、stream_done 刷出最终音频。

Scheduler 和 Stage 之间的接口极简：

```python
# Scheduler 只有这两个队列
inbox:  Queue[IncomingMessage]    # Stage 往里塞
outbox: Queue[OutgoingMessage]    # Scheduler 往里放，Stage 往外取
```

---

## 6. ModelRunner：真正接触模型的地方

Scheduler 管"什么时候算"，**ModelRunner 管"怎么算"**。这是唯一感知具体模型的一层。

它通过两个 hook 嵌入 sglang 的 forward 流程：

```
prepare_forward()  →  model.forward()  →  post_forward()
      ↑                                      ↑
   注入多模态 embedding                  捕获 hidden states / codebook 输出
```

Omni 有两类 ModelRunner：

- **ThinkerModelRunner**：`prepare_forward` 里注入图像/音频/视频的 embedding，然后正常 forward。
- **FeedbackARModelRunner**：用于 talker 这种"上一步输出是下一步输入"的 AR 模型。`write_buffers` 把上一步反馈写进模型 buffer，`extract_output` 抽取 codebook 结果。反馈闭环在同一个 ModelRunner 内。

---

## 7. 一张真正的调用链（以 Qwen3-Omni speech 为例）

```
thinker stage（GPU 0）：
  Stage 收 ZMQ SubmitMessage → relay 拿 payload → inbox
  → OmniScheduler: recv_requests → process_input_requests → get_next_batch_to_run
  → ThinkerModelRunner: prepare_forward(注入embedding) → forward → post_forward(捕获hidden states)
  → scheduler outbox

  两条路：
  ├→ normal path: hidden states 作为完整 payload → relay.write → ZMQ DataReady → talker stage
  └→ stream path: text token → decode stage（终端，输出文本）

talker stage（GPU 1）：
  Stage 收 DataReadyMessage → relay 拿 hidden states → inbox
  → OmniScheduler
  → FeedbackARModelRunner: write_buffers → forward → extract_output(codec codes)
  → outbox → relay stream chunk → ZMQ StreamMessage → code2wav stage

code2wav stage（GPU 1）：
  Stage 收 stream chunk → Code2WavScheduler 累积解码 → 输出音频波形
  → 终端完成 → Coordinator 合并 text + audio → HTTP 响应
```

---

## 8. 用 YAML 声明一条流水线

不需要看懂所有字段，核心结构就这些：

```yaml
relay_backend: shm            # 数据面后端
fused_stages:                 # 合并在同一进程的 stage（走 LOCAL_OBJECT）

stages:
  - name: preprocessing       # Stage 名字
    process: preprocessing    # 进程组
    factory: xxx.create_preprocessing  # 工厂函数（构造 scheduler/runner）
    next: [thinker]           # 静态下游

  - name: thinker
    gpu: 0                    # 绑 GPU 0
    next: [decode, talker]    # 可以多个下游
    stream_to: [talker]       # 流式发给 talker

  - name: decode
    terminal: true            # 终端 stage（next 和 terminal 二选一）

  - name: talker
    gpu: 1
    next: [code2wav]

  - name: code2wav
    gpu: 1
    terminal: true
```

关键规则：`next` 和 `terminal` 必须恰好声明一个；`stream_to` 声明流式下游；`wait_for` 声明 fan-in 上游（多路聚合时用）。

---

## 9. 学会这些就可以动手了

建议顺序：

1. 跑通 speech server，看启动日志里每个 stage 的进程和 GPU 绑定
2. 打开 `pipeline/stage/runtime.py`，看 Stage 主循环——这是理解一切的入口
3. 打开 `pipeline/control_plane.py`，看 ZMQ 消息是怎么收发路由的
4. 打开 `relay/shm.py`，看最简单的 relay 后端是怎么 put/get tensor 的
5. 打开 `scheduling/omni_scheduler.py`，对比 sglang 上游 Scheduler，看"复用了什么，替换了什么"
6. 打开 `models/qwen3_omni/`，看一个真实模型的 stage 拓扑是怎么配出来的

你想改代码时，最可能动的三个地方：加一个新 stage（写 factory + 配 YAML）、改 ModelRunner hook（加新的 forward 前后处理）、换 relay backend（改配置一行）。
