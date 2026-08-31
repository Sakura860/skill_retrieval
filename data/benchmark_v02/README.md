# benchmark_v02 数据说明

本扩展集专门为 `body-aware BM25`、两阶段 Schema disclosure/repair 和 typed Skill graph 构造，不替代冻结的 `benchmark_v01`。

## 范围

- 20 条新 Task：dev 10 条、confirmation/test 10 条；后者在方法冻结前禁止运行指标。
- 复用 v0.1 的 24 个真实 Skill、handler 和环境 fixture，不增加无关 Skill。
- 覆盖 `body_disambiguation`、`low_initial_rank`、`multi_skill_graph`、`prerequisite_completion`、`dataflow`、`alternative_conflict`、`schema_repair`。
- 每条 Task 都有结构化 `inputs`、真实 handler 和确定性 verifier。
- 每个模板只属于一个 split；方法冻结前只运行 dev。
- confirmation/test 覆盖文件、JSON、SQLite、计算和多步 dataflow 五个任务族；开发期只做结构和 verifier 完备性检查，不调用模型、不查看任务成功率。

## Graph 口径

`metadata.graph_edges` 是显式标注的诊断 graph：prerequisite 可用于候选补齐，dataflow 用于可执行链检查，alternative/conflict 只提供边界证据，不能自动扩张候选。Graph 任务同时记录 `intentionally_omitted_skill_ids`，用于比较原始候选与 prerequisite completion。

## 生成与验证

```powershell
python data/benchmark_v02/build_dataset.py
python tests/test_benchmark_v02.py
```

`metadata.raw_ranks` 保存 brief/detailed/all-field BM25 对每个 gold Skill 的完整池排名；`candidate_skill_ids` 仅供 Planner/graph 的固定候选实验使用，不能代替检索排名。
