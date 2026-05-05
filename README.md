# EvoAgent: 会自我生长的 Agent 系统

EvoAgent 是一个从 0 到 1 的深度工程化实现，核心能力由四个正交模块组成：

- 任务难度评估器（路由层）
- Sub-Agent 并行编排
- Skill 进化引擎（自动生成、沙盒验证、注册/淘汰）
- 三层记忆系统（工作/情节/语义）

## 1. 快速开始（Conda）

```bash
conda env create -f environment.yml
conda activate evoagent
pip install -e .
```

可选：复制 `.env.example` 为 `.env` 并填写密钥（用于 LLM，多供应商适配）。不要把 `.env` 提交到版本库。

```bash
# Provider routing
LLM_PROVIDER=deepseek
MODEL=deepseek-chat
FAST_MODEL=deepseek-chat
STANDARD_MODEL=deepseek-chat
REASONING_MODEL=deepseek-chat
LLM_BACKUP_MODELS=deepseek-chat

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Volcengine Ark / 火山方舟（使用 ep-... 或 ep-m-... endpoint id 时）
VOLCENGINE_API_KEY=your_ark_key
VOLCENGINE_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# Optional: other providers
OPENAI_API_KEY=
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

运行时状态默认写入 `data/`，包括 SQLite、Chroma 语义库、观测日志和自动生成的 Skill。它们属于本地运行产物，默认不应提交。

## 2. 运行

```bash
evoagent run "请帮我分析这个需求并给出方案"
evoagent chat
evoagent console
evoagent health
evoagent skills --status active --limit 10
evoagent semantic "并行编排最佳实践" --top-k 3
```

### 2.1 终端交互式 CLI UI

```bash
evoagent chat --session-id demo
```

进入后直接输入任务即可执行，不用每次重复敲 `evoagent run`。

- `/help` 查看快捷命令
- `/status` 查看当前状态栏信息
- `/new` 新建一个会话 id（自动生成）
- `/session <id>` 切换会话
- `/reset` 清空当前会话上下文
- `/history [n]` 查看最近对话卡片
- `/clear` 清屏并重绘 UI
- `/json on|off` 切换 JSON 输出
- `/exit` 退出

注：当前构建为 CLI-only，`evoagent ui` 已弃用并会提示改用 `evoagent chat`。

## 3. 核心设计映射

- 路由层：`src/evoagent/core/router.py`
- 编排层：`src/evoagent/core/orchestrator.py`
- Skill 引擎：`src/evoagent/skills/evolution.py`
- 记忆系统：`src/evoagent/memory/layers.py`
- 闭环学习：`src/evoagent/core/feedback.py`

## 4. 自动进化 Skill（已落地）

- 当任务评分较低、路径过长或置信度低时，会自动触发 Skill 进化。
- 新 Skill 会经过沙盒测试，测试通过后自动注册并持久化到 `data/skills_store`。
- 后续任务开始前会自动做 Skill 匹配，命中后优先执行已进化 Skill，形成“写出来就能用”的能力闭环。

## 5. 测试

```bash
pytest -q
```

如果在 WSL 中使用 Conda 环境，可直接运行：

```bash
conda run -n evoagent python -m pytest -q
```

检索链路端到端评测（含数据集）：

```bash
python scripts/evaluate_search_retrieval_e2e.py \
	--dataset data/observability/eval_search_e2e.json \
	--json-out data/observability/eval_search_e2e_result.json
```

## 6. 安全默认值

- `fetch_url_text` 默认只允许 `http/https`，并拒绝访问 localhost、私网和保留网段；确需访问内网时显式设置 `EVO_ALLOW_PRIVATE_NETWORK_FETCH=true`。
- 自动生成的 Skill 在注册前会做 AST 检查和 pytest 验证；运行时也会重新检查，并在隔离子进程中执行。
- Skill 运行默认超时为 10 秒，可通过 `SKILL_RUNTIME_TIMEOUT_SECONDS` 调整。
