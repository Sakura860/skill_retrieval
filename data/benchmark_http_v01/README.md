# benchmark_http_v01

这是用于检验既有 `adaptive_signals` 跨任务族迁移的本地 HTTP/API 数据集。该策略形成时只见过计算、JSON 转换、文件、SQLite 和多步 dataflow，从未见过本数据集、HTTP 方法、状态码、Bearer header 或幂等键。

- 9 个 HTTP Skill，均由真实 loopback HTTP handler 执行，没有 mock executor。
- 12 条任务：dev 6、confirmation 6；两个 split 的实体、路径和 template ID 完全分离。
- 每条任务启动独立的 `127.0.0.1` 随机端口服务，fixture 不访问公网。
- `http_state` verifier 同时检查最终输出、方法、路径、JSON body、相关 header、响应状态和精确调用次数。
- 两个 split 都覆盖单 Skill 具名输入、token→Bearer 多 Skill 数据流，以及“限定成功状态、原样错误正文、不得自动重试”的行为约束。

重新生成：

```powershell
python data/benchmark_http_v01/build_dataset.py
```

生成后必须运行 `python tests/test_benchmark_http.py`。confirmation 只能在冻结新协议后由独立 registry 预留并逻辑运行一次；不得使用其结果修改 `configs/disclosure_policy_v01.json`。
