# Skill Agent Task 5 阶段完成总结（2026-08-24）

## 1. 当前结论

本阶段已经完成 body-aware 检索、两阶段 Planner、Schema 校验与一次修复、typed Skill graph、定向 benchmark v0.2、真实 handler/verifier 闭环，以及 Planner/graph 的三轮 DeepSeek dev 稳定性实验。

当前最终组合候选为：

```text
detailed BM25 top-10
  → typed graph（只扩张 prerequisite，最多 +2）
  → two-stage Planner
  → 参数/数据流 Schema 校验
  → 最多一次 repair
  → 隔离真实 handler
  → 确定性 verifier
```

该组合已完成第一轮 `benchmark_v02` dev：10/10 Task Success，0 API failure。冻结要求是三轮完整 dev，因此当前配置仍是 `candidate`，没有运行 held-out test。

## 2. 已完成的工程实现

### 2.1 Body-aware BM25

- `Skill.to_text` 支持 `brief`、`detailed`、`all` 三档检索文本。
- `all` 稳定序列化名称、简介、详细说明、参数/返回 Schema、tags、examples、dependencies 和 metadata。
- BM25 对 `text_level` 做显式校验；查询词顺序稳定，避免跨进程近似并列排名波动。
- 新增离线比较入口 `experiments/run_task5_retrieval.py`，可选择 v0.1/v0.2 与 dev/test split。

### 2.2 两阶段 Planner 与执行前校验

- 保留 `one_stage` 基线，新增 `two_stage` 模式。
- 第一阶段只看 brief 和稳定 Skill ID，输出“原子需求 → Skill ID”覆盖证据。
- 第二阶段只加载已选 Skill 的完整 body/Schema，再生成参数和顺序。
- 新增 `Task.inputs` 与 `$input.<field>` 精确绑定；保留 `$task` 和 `$last_output`。
- 新增参数 Schema 校验：required、type、enum、items、范围、additionalProperties。
- 新增 `$last_output` 返回 Schema → 下一步参数 Schema 兼容性检查。
- 最多允许一次 Schema-based repair；修复后仍不合法则禁止执行。
- Planner 可在发现冗余或不兼容步骤时删除该步骤，并把排序/过滤下推给上游 Skill。
- 评测记录 selection evidence、Planner 调用数、repair 次数、校验错误及分阶段上下文 Token。

### 2.3 Typed Skill graph

- 定义 `prerequisite`、`dataflow`、`co_use`、`alternative`、`conflict` 五种关系。
- 只有 `prerequisite` 可以自动补充候选；`alternative/conflict/co_use` 不得作为硬依赖扩张。
- 支持最大补点数、缺失依赖、环、扩张上限及稳定前置顺序诊断。
- `GraphOrganizer` 确保补入 Skill 同时进入 Planner 可见集合和 Executor 白名单。
- 支持根据返回/参数 Schema 推断诊断性 dataflow 边。

### 2.4 数据与评测闭环

- 新建独立 `benchmark_v02`，不修改 v0.1 已冻结的假设与 test。
- v0.2 共 14 条 Task（dev 10/test 4），覆盖 7 类研究问题：body disambiguation、低初始排名、multi-Skill graph、prerequisite completion、dataflow、alternative/conflict、Schema repair。
- 复用 24 个真实 Skill、真实 handler 和隔离环境 fixture，没有用无关 Skill 扩充样本。
- 全部任务包含结构化 inputs、确定性 verifier，并保持模板跨 split 隔离。
- v0.1 的数组 item Schema 已补齐；`tjson07`、`tfile03` 的结构化输入和精确换行契约已固化为回归。
- DeepSeek runner 支持逐任务 checkpoint、API/任务失败分离、指数退避重试和续跑。
- 修复 Windows curl 回退把 UTF-8 JSON 按 GBK 解码的问题。

## 3. 实验结果

所有模型实验使用 DeepSeek V4-Pro、temperature 0、thinking disabled。以下只使用 dev；test 尚未执行。

### 3.1 Body-aware retrieval（benchmark_v02 dev，10 Task）

| 索引字段 | Hit@1 | Recall@3 | Recall@5 | MRR | 平均最佳 gold 排名 |
|---|---:|---:|---:|---:|---:|
| brief | 40.00% | 40.00% | 75.00% | 0.5318 | 4.40 |
| detailed | 100.00% | 95.00% | 95.00% | 1.0000 | 1.00 |
| all-field | 100.00% | 90.00% | 95.00% | 1.0000 | 1.00 |

结论：收益主要来自详细功能边界；把所有 Schema/metadata 都加入索引没有进一步提高 Hit@1/MRR，并在多 Skill Recall@3 上略有干扰。因此最终候选选择 `detailed`，不宣称 all-field 总是更好。

### 3.2 Planner 三轮稳定性（benchmark_v01 dev，18 Task × 3 方法 × 3 轮）

三轮共 162 个任务级模型观测，0 API failure。

| 方法 | Task Success 均值 ± SD | Selection F1 | Sequence | 平均 Skill Context | 总 Token/轮均值 ± SD | 稳定任务数 |
|---|---:|---:|---:|---:|---:|---:|
| one-stage | 72.22% ± 0.00% | 100.00% | 100.00% | 807.50 | 14,479.3 ± 32.6 | 18/18 |
| two-stage，无 repair | 92.59% ± 3.21% | 92.59% | 92.59% | 541.67 | 15,423.7 ± 20.4 | 17/18 |
| two-stage，最多一次 repair | 100.00% ± 0.00% | 100.00% | 100.00% | 541.67 | 15,828.0 ± 32.0 | 18/18 |

关键观察：

- one-stage 三轮都失败在同一 5 条任务：`tcalc06`、`tjson01`、`tjson03`、`tjson06`、`tdb03`。
- no-repair 的 `tjson03` 三轮都失败；`tjson06` 出现 1 次失败、2 次成功，是唯一不稳定任务。
- one-repair 三轮全部 18/18，通过一次受 Schema 约束的修复消除了参数生成脆弱点。
- 相比 one-stage，one-repair 的 Skill Context 降低 32.92%；代价是通常需要两次 Planner 调用，总模型 Token 约增加 9.31%（按三轮均值计算）。

### 3.3 Typed graph 三轮诊断（benchmark_v02 dev，2 Task × 2 方法 × 3 轮）

三轮共 12 个任务级模型观测，0 API failure。

| 方法 | Task Success 均值 ± SD | Selection F1 均值 ± SD | Sequence | 平均 Skill Context | 总 Token/轮均值 ± SD |
|---|---:|---:|---:|---:|---:|
| 不补 prerequisite | 0.00% ± 0.00% | 41.67% ± 14.43% | 0.00% | 1,919.50 | 3,162.0 ± 20.5 |
| 补 prerequisite，最多 +2 | 100.00% ± 0.00% | 100.00% ± 0.00% | 100.00% | 2,122.50 | 3,473.7 ± 4.6 |

开启补齐后三轮都只加入预期节点：`v2graph01 → sfile_copy`、`v2graph02 → sfile_create`。关闭组的 Selection F1 有模型波动，但 Task Success 和 Sequence 的因果对照稳定。

### 3.4 最终组合候选首轮（benchmark_v02 dev，真实 retriever top-10）

| 指标 | 结果 |
|---|---:|
| 完成任务 / API failure | 10 / 0 |
| Task Success | 100.00% |
| Selection F1 | 100.00% |
| Sequence Accuracy | 100.00% |
| Retrieval MRR | 1.0000 |
| Recall@1 / @3 / @5 / @10 | 90% / 95% / 95% / 100% |
| 平均 Skill Context | 494.20 |
| 总 Token | 8,777 |
| 平均 Planner 调用 | 2.00 |
| 平均 repair | 0.00 |

有效性边界：本轮 detailed top-10 已直接召回两个 graph prerequisite，因此没有触发自动补点；graph completion 的有效性证据来自上面的独立三轮候选遗漏 off/on 诊断。本轮只能证明三组件组合没有造成回归，不能替代 graph 因果实验。

## 4. 验证状态

已完成：

- 全项目 `compileall`、`git diff --check`。
- 当前 13 个测试脚本全部通过，包括 Agent 执行、DeepSeek 参数、真实 handler E2E、检索指标、v0.2 数据审计和 typed graph。
- v0.2 的 10 条 dev Task 已通过 Scripted LLM + 真实 handler + verifier。
- 新增聚合器和最终候选 runner 已通过 Python 编译。
- 冻结保护已实测：配置状态为 `candidate` 时运行 `--split test` 会立即拒绝，且不生成结果文件。
- 被中断的最终候选首轮已经自然完成并保存完整 checkpoint；没有残留后台 Python 进程。

尚未完成：

- 最终组合候选只完成 1/3 轮 dev，尚需两轮稳定性复跑。
- 配置 `configs/task5_candidate_20260824.json` 尚未改为 `frozen`。
- `benchmark_v01` 和 `benchmark_v02` 的 held-out test 均未运行。
- graph 的 held-out test 没有专门 prerequisite-omission 样本；当前 test 只能检查 soft relation 不误扩张，不能独立复验补齐收益。
- 大规模 Skill 库延迟/内存、双塔训练、外部任务环境成功率仍未验证。

## 5. 关键产物

| 文件 | 用途 |
|---|---|
| `agent/schema_validation.py` | 参数和 `$last_output` 数据流 Schema 校验 |
| `organization/typed_graph.py` | typed graph 数据结构与扩张逻辑 |
| `data/benchmark_v02/` | 14 条定向 Task、Skill 和环境 fixture |
| `experiments/run_task5_retrieval.py` | brief/detailed/all-field 离线检索对比 |
| `experiments/run_task5_planning.py` | Planner 三方法受控 DeepSeek runner |
| `experiments/run_task5_graph.py` | prerequisite completion 独立 off/on runner |
| `experiments/analyze_task5_repeats.py` | 跨重复均值、SD、逐任务稳定性聚合 |
| `experiments/run_task5_candidate.py` | 最终组合候选 runner 与 held-out 冻结保护 |
| `configs/task5_candidate_20260824.json` | 当前未冻结的最终组合候选配置 |
| `results/task5_dev_repeat_summary_20260824.json` | Planner/graph 三轮完整聚合数据 |
| `results/task5_candidate_v02_dev_repeat1.json` | 最终组合候选首轮完整证据 |

## 6. 下次恢复入口

保持当前代码、prompt、Schema、数据和配置不变，继续两轮 dev：

```powershell
python experiments/run_task5_candidate.py --split dev --output results/task5_candidate_v02_dev_repeat2.json --max-retries 2
python experiments/run_task5_candidate.py --split dev --output results/task5_candidate_v02_dev_repeat3.json --max-retries 2
```

两轮完成后先聚合并检查逐任务结果、Token 和 API failure。只有三轮整体稳定，才把候选配置状态改为 `frozen`，记录冻结时间和固定文件摘要，然后首次且只运行一次 held-out test。当前工作树未提交，恢复时应保留现有修改和结果文件。
