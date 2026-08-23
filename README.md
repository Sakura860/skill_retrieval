# skill-agent

面向大规模 Skill 库的检索、组织、调用与评测框架。

## 运行流程

```text
Task
  → 多层检索（粗略描述召回 + 详细描述重排）
  → Skill 组织（扁平 / 分层披露 / 依赖图）
  → 结构化规划（技能、参数、顺序）
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

## 执行函数注册

数据文件只描述 Skill，不保存可执行源码。执行函数必须由宿主程序显式注册：

```python
def calculator(expression: str) -> str:
    return str(your_safe_calculator(expression))

agent = Agent(
    llm=llm,
    organizer=organizer,
    skill_handlers={"calculator": calculator},
)
```

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

`Skill Context Tokens` 使用本地中英文估算器，便于比较组织策略；`Total Tokens` 优先采用 API 返回的真实 usage。Mock 模式使用同一估算器。

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
