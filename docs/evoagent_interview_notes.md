# EvoAgent 项目面试讲述文档

## 1. 项目一句话介绍

EvoAgent 是一个基于 Python 的多 Agent 智能任务执行系统。它可以把用户任务拆分成多个步骤，再由不同角色的 Agent 分别执行，最后汇总结果并返回给用户。

这个项目主要用到了 FastAPI、LangGraph、SQLModel、SQLite、LiteLLM 和 Pytest。

面试时可以这样说：

> 我做的是一个多 Agent 智能任务执行系统，主要目标是让系统能自动拆解任务、调用不同类型的子 Agent 执行，并记录每次任务的执行过程，方便后续调试和扩展。

## 2. 我主要做了什么

我在项目中主要负责几部分：

1. 设计任务运行数据模型，比如 Run、Step、Artifact、ToolCall。
2. 重构 LLM 调用模块，把模型路由、健康检查、失败重试和工具调用拆开。
3. 使用 LangGraph 搭建 Agent 工作流，包括 Planner、Worker、Reviewer、Reporter。
4. 使用 FastAPI 提供任务创建、状态查询、步骤查询和结果查询接口。
5. 编写 Pytest 测试，覆盖核心模块。

面试时可以这样说：

> 我主要负责系统的运行时和 LLM 调用层。运行时部分负责记录一次任务从开始到结束的完整过程；LLM 调用层负责选择模型、处理失败重试和工具调用。这样系统既能执行任务，也能追踪任务每一步发生了什么。

## 3. 模块一：任务运行数据模型设计

### 设计了哪些表

我设计了四个核心模型：

- Run：表示一次完整任务
- Step：表示任务中的一个执行步骤
- Artifact：表示执行过程中产生的结果或文件
- ToolCall：表示一次工具调用记录

它们的关系大概是：

```text
Run
 ├── Step
 │    └── ToolCall
 └── Artifact
```

### 为什么这样设计

因为一个 Agent 任务不是简单的一次函数调用，而是一个有过程的执行链路。比如用户问一个复杂问题，系统可能要：

1. 先规划任务
2. 再调用搜索 Agent
3. 再调用代码 Agent
4. 最后汇总答案

如果只保存最终答案，后面很难知道中间哪一步失败了。所以我把任务过程拆成 Run 和 Step，再把中间产物和工具调用单独记录。

### 遇到的问题

一开始比较容易只关注最终结果，但实际调试时发现，只看 final answer 不够。如果结果不对，需要知道：

- 是哪个 Agent 出错了
- 哪个工具调用失败了
- 每一步的输入输出是什么
- 任务是成功、失败还是取消

### 解决方式

我把运行过程持久化到数据库中，并给每个步骤加上状态字段：

```text
pending / running / succeeded / failed / skipped
```

这样后续无论是 API 查询，还是前端展示，都可以清楚看到任务执行链路。

## 4. 模块二：LLM 调用模块重构

### 原来的问题

原来的 LLMClient 职责比较多，一个类里同时处理：

- 模型选择
- Provider 认证
- LiteLLM 调用
- 失败重试
- 模型健康检查
- 工具调用

这样代码会比较难维护，后面如果要接新模型或者改重试策略，容易影响其他逻辑。

### 我的设计

我把 LLM 模块拆成几个部分：

```text
llm/
  health.py      # 记录模型健康状态
  router.py      # 负责选择候选模型
  adapter.py     # 负责调用 LiteLLM
  tools.py       # 负责工具调用循环
  service.py     # 对外统一入口
```

对外仍然保持原来的调用方式：

```python
llm = LLMClient()
answer = llm.generate("hello")
```

内部实际走的是新的 LLMService。

### 为什么这样设计

这样拆分以后，每个模块职责更清楚：

- Router 只管选模型
- HealthStore 只管模型健康状态
- Adapter 只管调用 LiteLLM
- ToolCallRunner 只管工具调用
- Service 负责串起来

好处是后续扩展更简单，比如增加一个模型 Provider，主要改 Adapter 和 Router，不需要改所有调用方。

### 遇到的问题

最大的问题是不能破坏旧代码。项目里很多地方已经写了：

```python
from evoagent.core.llm import LLMClient
```

如果直接改类名，会导致大量代码需要一起改。

### 解决方式

我保留了兼容层：

```python
LLMClient = LLMService
```

这样旧代码不用改，但内部实现已经切换到新的结构。

## 5. 模块三：LangGraph 工作流设计

### 设计了哪些节点

我用 LangGraph 设计了几个基础节点：

- Planner：负责生成任务计划
- Supervisor：决定下一步调用哪个 Worker
- Worker：执行具体任务
- Reflection：可选的反思和修正节点
- Reporter：汇总最终答案

执行流程是：

```text
Planner -> Supervisor -> Worker -> 是否继续
                         ├── 继续执行下一个 Worker
                         ├── 进入 Reflection
                         └── Reporter 输出结果
```

### 为什么使用 LangGraph

普通函数调用适合简单流程，但多 Agent 系统会有状态流转，比如当前执行到第几步、每一步输出是什么、是否需要反思、是否可以恢复任务。

LangGraph 更适合这种有状态的 Agent 流程，可以把每个节点拆开，也方便后续扩展。

### 遇到的问题

一个坑是工作流状态需要统一管理。如果每个节点随便返回字段，后面节点就很容易拿不到需要的数据。

### 解决方式

我定义了统一的 AgentState，里面包含：

- run_id
- user_query
- plan
- steps
- current_step
- worker_outputs
- final_answer
- confidence
- metadata

这样每个节点都围绕同一个状态对象读写，流程更稳定。

## 6. 模块四：FastAPI 网关设计

### 提供了哪些接口

我设计了几个基础接口：

- POST /runs：创建任务
- GET /runs/{run_id}：查询任务状态
- GET /runs/{run_id}/steps：查询任务步骤
- GET /runs/{run_id}/artifacts：查询任务产物
- GET /health：健康检查

### 为什么要加 API

原来系统主要是 CLI 调用，不方便接前端或外部系统。加 FastAPI 后，可以把 Agent 系统变成一个服务，对外通过 HTTP 提供能力。

### 遇到的问题

任务执行可能比较耗时，如果 HTTP 请求一直阻塞，用户体验会不好。

### 解决方式

创建任务时只返回 run_id 和初始状态，然后后台执行任务。用户可以通过查询接口查看任务状态和执行步骤。

## 7. 模块五：测试设计

### 测试覆盖了什么

我写了 Pytest 测试，主要覆盖：

- 数据模型创建和状态更新
- LLM 模块的路由和健康检查
- LangGraph 节点是否能正常运行
- Worker 返回格式是否正确
- FastAPI 接口是否能返回正确结果

### 为什么要 mock LLM

LLM 调用依赖外部 API，不适合在单元测试里真实请求。真实请求会带来几个问题：

- 需要 API Key
- 网络不稳定
- 成本不可控
- 返回结果不稳定

所以测试里主要 mock LLM 返回值，只测试系统逻辑。

## 8. 面试中可以重点强调的点

这个项目虽然是实习项目，但可以重点讲三个能力：

### 1. 工程拆分能力

我不是把所有逻辑写在一个类里，而是按职责拆分模块，比如 LLM 调用层拆成 Router、HealthStore、Adapter、ToolRunner。

### 2. 可观测性意识

我设计了 Run、Step、Artifact、ToolCall，用来记录任务执行全过程。这样不仅能得到最终答案，还能知道中间每一步发生了什么。

### 3. 测试意识

我给核心模块写了单元测试，并且通过 mock 避免真实 API 调用，让测试更稳定。

## 9. 面试回答模板

如果面试官问：“你这个项目里最有技术含量的部分是什么？”

可以回答：

> 我觉得比较有技术含量的是 LLM 调用层和任务运行时的设计。原来的 LLMClient 职责比较重，我把它拆成模型路由、健康检查、Provider 适配和工具调用几个模块，同时保留原来的 LLMClient 接口不变，避免影响旧代码。任务运行时方面，我设计了 Run、Step、Artifact、ToolCall 这些模型，记录一次任务从创建、执行到完成的全过程，这样后续排查问题时可以看到每一步的输入、输出和状态。

如果面试官问：“你遇到过什么问题？”

可以回答：

> 一个问题是重构 LLM 模块时不能影响已有调用方，因为项目里很多地方已经依赖 LLMClient。如果直接改接口，会导致大量代码要一起改。我最后用兼容层解决，让 LLMClient 指向新的 LLMService，这样外部调用不变，内部结构完成了解耦。另一个问题是测试不能依赖真实 LLM API，所以我在测试里 mock LiteLLM 调用，只验证路由、失败重试和状态更新逻辑。

如果面试官问：“为什么要设计 Run、Step 这些表？”

可以回答：

> 因为多 Agent 系统的执行过程比较长，如果只保存最终答案，出现问题时很难定位。所以我把一次任务抽象成 Run，把每个执行阶段抽象成 Step，再用 Artifact 保存中间产物，用 ToolCall 保存工具调用记录。这样可以完整追踪一次任务的执行链路。

## 10. 简历对应描述

可以放在简历上的简化版本：

```text
EvoAgent 多 Agent 智能任务执行系统
技术栈：Python、FastAPI、LangGraph、SQLModel、SQLite、LiteLLM、Pytest

- 参与开发一个多 Agent 智能任务执行系统，支持任务拆解、子 Agent 执行、结果汇总和错误处理。
- 设计 Run、Step、Artifact、ToolCall 等数据模型，用于记录任务状态、执行步骤和工具调用结果。
- 重构 LLM 调用模块，拆分模型路由、健康检查、失败重试和工具调用逻辑，提高代码可维护性。
- 使用 LangGraph 搭建 Agent 工作流，实现 Planner、Worker、Reviewer、Reporter 等节点的基础调度。
- 基于 FastAPI 提供任务创建、状态查询、步骤查询和结果查询接口。
- 编写 Pytest 单元测试，覆盖数据模型、LLM 模块、工作流节点和 API 接口。
```
