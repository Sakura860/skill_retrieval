# skill-agent

面向大规模 Skill 库的检索、组织、调用与评测框架。

## 运行流程

```text
Task
  → 可控粒度检索（brief / detailed / all-field）
  → Skill 组织（扁平 / 分层披露 / 依赖图）
  → 一阶段规划，或 brief 选择后按需加载 Schema 的两阶段规划
  → 注册函数执行
  → 反思与任务级评测
```

## 快速开始

```powershell
cd skill-agent
pip install -r requirements.txt
```

打开项目根目录的 `.env`，填写 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=你的_API_Key
```

运行默认实验：

```powershell
python main.py
```

默认配置位于 `configs/default.yaml`。主入口会读取其中的 LLM、检索器、组织器、Top-K 和 Agent 参数。
如需运行 Embedding 或双塔训练，额外安装：

```powershell
pip install -r requirements-ml.txt
```

### DeepSeek V4

默认模型为 `deepseek-v4-pro`，并关闭思考模式，便于稳定比较不同 Skill 检索和组织策略：

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-pro
  temperature: 0.0
  thinking: disabled
  reasoning_effort: high
```

低成本批量实验可改为 `deepseek-v4-flash`。启用思考模式时：

```yaml
thinking: enabled
reasoning_effort: high  # high | max
```

思考模式下不会发送 `temperature`；`reasoning_effort` 只在思考模式下发送。

项目只支持两种运行模式：

```python
LLM(provider="deepseek")  # 真实实验
LLM(provider="mock")      # 离线测试
```

`openai` 包仅作为 DeepSeek 官方 OpenAI 兼容协议的客户端使用，项目不提供 OpenAI 或 Anthropic 模型入口。若运行环境暂时无法安装该包，DeepSeek 调用会回退到同一兼容 HTTP 接口；在仅系统 curl 网络通道可用的 Windows 环境中再安全回退到 curl，API Key 通过标准输入传递，不出现在进程参数中。

## Skill 数据结构

Skill 同时提供粗粒度和细粒度信息：

| 字段 | 用途 |
|---|---|
| `id` | 稳定标识与评测标签 |
| `name` | 规划和函数注册使用的名称 |
| `brief_description` | 第一阶段召回与候选概览 |
| `detailed_description` | 重排、参数选择和详细披露 |
| `category` | 层次分组 |
| `parameters` | 输入 JSON Schema |
| `returns` | 返回值 Schema |
| `tags` | 检索补充词 |
| `examples` | 用法与语义补充 |
| `dependencies` | 前置 Skill ID |
| `metadata` | 风险、权限等扩展属性 |

旧版数据中的 `description` 和 `metadata.depends_on` 仍可由 loader 转换。

`BM25Retriever(text_level=...)` 支持三种可归因设置：`brief` 使用名称、类别、简介和 tags；`detailed` 增加详细功能边界；`all` 使用包含参数/返回 Schema、示例、依赖和 metadata 的完整 Skill body。

已发表方法对照通过 `retrieval.published` 接入官方 SkillRouter 0.6B encoder 和 reranker。适配器固定上游 commit、官方模型 ID、query/document/prompt 格式和本项目字段映射，缺少 `torch/transformers` 或权重时显式失败，绝不回退到 hash。统一入口：

```powershell
python experiments/run_published_baselines.py --dataset benchmark_v02 --split dev `
  --methods bm25_brief bm25_all skillrouter_embedding skillrouter_pipeline `
  --output results/published_baselines.json
```

官方模型依赖单独列在 `requirements-published.txt`，避免污染 Mock/离线测试环境。当前已在 CPU/float32 下完成官方权重实测；最终可引用的是 v02 严格 top-20 结果，模型准备、索引与在线查询分别计时。汇总、有效性边界与证据 SHA 见 `results/published_baseline_and_confirmation_report_20260830.md`，不要使用报告中标为 superseded 的早期 pipeline 文件。

## 执行函数注册

数据文件只描述 Skill，不保存可执行源码。执行函数必须由宿主程序显式注册：

```python
def calculator(expression: str) -> str:
    return str(your_safe_calculator(expression))

agent = Agent(
    llm=llm,
    organizer=organizer,
    skill_handlers={"calculator": calculator},
    planner_mode="two_stage",
    max_argument_repairs=1,
)
```

`two_stage` 的第一次模型调用只从 brief 中选择 Skill，第二次按 `planner_disclosure_level` 加载已选 Skill 的 `brief`、`schema` 或 `full` 定义来生成顺序和参数。`adaptive` 是模型自报层级的失败对照；`adaptive_signals` 使用开发集失败归纳出的可审计策略：单 Skill 且具名输入覆盖必填参数时用 brief，多 Skill 数据流用 schema，命中已冻结的非 Schema 行为约束时用 full。逐任务结果记录判据、命中信号和任何升级原因。

参数在执行前按 JSON Schema 校验；若缺少必填字段、类型/枚举/范围不符，可进行最多一次带错误路径的修复。adaptive 从 brief 失败时只升级到 schema；修复后仍不合法则不会进入 Executor。默认 `one_stage` 保留为对照基线。

Task 可通过公开给 Planner、但不包含 verifier 答案的 `inputs` 保存精确业务输入。计划参数用 `$input.<字段>` 引用，适合 JSON 对象、数组及换行敏感文本；`$task` 仍表示整条自然语言指令，`$last_output` 表示动态上一步结果。

## Typed Skill graph

`TypedSkillGraph` 区分五类边：prerequisite 来自 `dependencies` 或显式边；dataflow 可由返回/输入 Schema 兼容性诊断；co-use 来自任务共现统计；alternative 和 conflict 来自 Skill metadata 或人工功能边界。只有 prerequisite 会自动扩张候选，避免把软相关或替代 Skill 错当成执行前置。

`GraphOrganizer(catalog=..., max_additional_skills=N)` 会在受控上限内递归补齐 prerequisite，按依赖顺序披露，并记录 `added_skill_ids`、缺依赖、环和扩张上限诊断。补充 Skill 只有在实际出现在上下文后才允许执行。

未注册函数、未知技能、缺少必填参数或执行异常都会记为失败。项目不再通过 `exec()` 运行数据中的任意代码。

`benchmark_v01` 使用统一 `SkillRegistry` 注册 24 个确定性 handler。每条任务在独立 `TaskEnvironment` 中 setup，文件访问只能使用受控相对路径，SQLite 写操作使用参数化语句，读查询使用只读连接；Agent 返回运行前后状态快照后环境立即 teardown。`execution_success` 只表示调用有效，最终仍必须由 Task verifier 判定 `task_success`。

## 端到端指标

| 类型 | 指标 | 计算口径 |
|---|---|---|
| 准确性 | **Skill Selection F1** | 每个任务对“已调用 Skill 集合”和标注集合计算 F1，再宏平均 |
| 准确性 | **Sequence Accuracy** | 已调用 Skill ID 序列与标注序列完全一致的任务比例 |
| 最终效果 | **Task Success Rate** | 通过 ground truth 或外部 verifier 的任务数 / 可评分任务数 |
| 效率 | **Avg. Skill Calls** | 每个任务的平均 Skill 调用次数 |
| 效率 | **Redundant Call Rate** | 超出标注序列所需次数的调用数 / 总调用数 |
| 上下文 | **Skill Context Tokens** | 每个任务 Skill 上下文的平均估算 Token 数 |
| 成本 | **Total Tokens** | 一次评测中 LLM 的 Prompt 与 Completion Token 总和 |
| 效率 | **Avg. Execution Steps** | 每个任务平均“规划步骤 + Skill 执行步骤”数 |
| 成本 | **Avg. Planner Calls** | 每个任务的平均规划模型调用数 |
| 成本 | **Avg. Argument Repairs** | 每个任务的平均参数修复调用数 |
| 延迟 | **LLM / End-to-end p50, p95** | 逐次记录 selection、planning、repair、reflection 的 wall-clock，并汇总任务端到端分位数 |

`Skill Context Tokens` 使用本地中英文估算器，便于比较组织策略；`Total Tokens` 优先采用 API 返回的真实 usage。Mock 模式使用同一估算器。LLM 调用轨迹只保存阶段、模型、token、字符数、耗时和错误类型，不保存 prompt 正文。

任务有旧版 `ground_truth` 时会自动使用精确匹配验收。新任务通过私有的 `evaluation` 字段指定确定性 verifier；该字段不会进入 Planner 提示词：

```json
{
  "id": "t001",
  "instruction": "计算结果并写入 result.json",
  "evaluation": {
    "verifier_type": "file_state",
    "expected_state": {
      "files": {
        "result.json": {"exists": true, "json": {"total": 231}}
      }
    },
    "required_effects": ["result.json"],
    "forbidden_effects": ["source.json"]
  }
}
```

内置 verifier 包括 `exact_match`、`json_match`、`file_state` 和 `sqlite_state`，统一接收任务、初始状态、最终状态、Agent 输出和执行轨迹，并返回可解释的 `VerifierResult`。复杂外部环境仍可通过 verifier registry 扩展；旧版 `success_evaluator` 参数保留兼容。没有 ground truth 或 verifier 的任务标记为 `unscored`，不会因为 handler 正常返回就计入成功率。

`run_benchmark()` 在同一次检索结果上计算检索指标并执行 Agent，返回可直接序列化的分层结果：

```json
{
  "run_info": {
    "run_id": "...",
    "task_count": 30,
    "scored_task_count": 26,
    "unscored_task_count": 4
  },
  "config": {
    "retriever": "BM25Retriever",
    "organizer": "HierarchicalOrganizer",
    "top_k": 5
  },
  "metrics": {
    "retrieval": {
      "recall@1": 0.65,
      "recall@5": 0.92,
      "mrr": 0.76,
      "ndcg@5": 0.84
    },
    "agent": {
      "skill_selection_f1": 0.81,
      "sequence_accuracy": 0.63,
      "task_success_rate": 0.57
    },
    "efficiency": {
      "avg_skill_context_tokens": 960.0,
      "total_tokens": 2400,
      "avg_skill_calls": 2.3
    }
  },
  "per_task": []
}
```

逐任务结果同时保留原始 BM25 排名、固定候选顺序、实际暴露/详细披露/裁剪的 Skill ID、上下文预算与实际 Token、调用序列、初末状态、`execution_success`、`task_success`、verifier 类型和失败原因，便于判断失败发生在候选、披露、规划、执行还是验收阶段。

## 实验

```powershell
python experiments/run_experiment.py
```

脚本比较 BM25、Embedding、多层检索与三种组织策略。默认使用 DeepSeek，完整消融会产生多次 API 调用。每次实验会在 `results/` 下保存包含配置、聚合指标和逐任务证据的 JSON。离线测试请显式传入 `LLM(provider="mock")`。

body-aware BM25 的确定性 dev/test 排名比较不会调用 LLM：

```powershell
python experiments/run_task5_retrieval.py --split dev
```

该脚本在相同 Skill 池、Task、BM25 参数和 split 上比较 `brief`、`detailed`、`all`，报告 Hit@1、Recall@K、MRR、NDCG@K、平均正确 Skill 排名和逐任务排名证据。

Planner 的真实 DeepSeek 受控对比：

```powershell
python experiments/run_task5_planning.py --task-ids tjson07,tfile03
```

脚本比较 one-stage、two-stage/no-repair 和 two-stage/one-repair，固定 brief-BM25 候选、Task context budget、温度与隔离初始环境，并保存逐任务选择覆盖、参数校验、Token 和 verifier 证据。

`benchmark_v02` 位于 `data/benchmark_v02/`，包含 14 条专用于 body、Schema repair 和 typed graph 的新任务。生成和审计：

```powershell
python data/benchmark_v02/build_dataset.py
python tests/test_benchmark_v02.py
```

typed graph prerequisite completion 的独立 dev 诊断：

```powershell
python experiments/run_task5_graph.py
```

当前 Task 5 的受控 dev 主表与失败归因见 `results/task5_dev_analysis_20260824.md`，三轮稳定性见 `results/task5_disclosure_dev_preregistered_v07_summary.json` 和 `results/task5_end_to_end_dev_v01_summary.json`。真实 BM25 top-10 协议已冻结并完成唯一一次 confirmation；结果登记、原始 SHA 与浮点边界审计见 `results/published_baseline_and_confirmation_report_20260830.md`。该确认集已经消费，禁止再次运行或据其结果修改 `adaptive_signals`。

导师反馈后的 Flat vs Hierarchical 定向数据位于 `data/benchmark_v01/`，包含 24 个 hard-negative Skill、34 条主任务和 10 条独立 Graph 诊断任务。数据假设、受控候选排名、dev/test 划分及当前使用边界见该目录的 `README.md`。任务 4 实验入口会直接消费每条任务固定的 BM25 候选顺序与预算：

```powershell
python experiments/run_task4_organization.py --split dev --repeats 3
```

脚本只比较 Flat 与 Hierarchical，逐条创建隔离环境，并保存配置快照、逐任务 JSONL、聚合 JSON/CSV、失败案例和 Graph 结构诊断。已完成的 108 个真实 DeepSeek 观测及结论见 `results/task4_organization_20260823T022006Z/analysis.md`；test split 保留到方案冻结后再运行。

任务 3 的四任务族真实 DeepSeek smoke：

```powershell
python experiments/run_task3_smoke.py
```

该脚本只跑四条单步 dev 任务、关闭 Reflection，并将完整结果保存到 `results/`；真实 API 不会在单元测试中调用。

## 测试

```powershell
python tests/test_smoke.py
python tests/test_benchmark_dataset.py
python tests/test_task_metrics.py
python tests/test_execution_runtime.py
python tests/test_real_handlers_e2e.py
python tests/test_retrieval_metrics.py
python tests/test_benchmark_output.py
python tests/test_agent_execution.py
python tests/test_task4_experiment_controls.py
python tests/test_deepseek_config.py
```

测试使用 Mock LLM，不调用外部 API。
