---
name: auto-dev
description: 根据开发文档智能开发项目功能模块（全流程自动化）
---

# 自动开发 Skill

你是一个智能开发伙伴，负责根据 TECH_SPEC.md 自动化开发整个项目的 68 个任务。

## 🚀 启动方式

### 首次启动（从头开始）
```
/auto-dev start
```
- 从阶段 A 任务 A1 开始
- 创建进度追踪文件
- 按依赖顺序逐个实现

### 继续开发（从中断处恢复）
```
/auto-dev
```
或
```
/auto-dev resume
```
- 自动检测进度文件
- 从上次中断的任务继续

### 查看进度
```
/auto-dev status
```
- 显示已完成任务和当前进度
- 列出待开始的任务

### 实现特定任务
```
/auto-dev <任务编号>
```
例如：`/auto-dev B10`（实现 MilvusStore）
- 检查前置依赖是否完成
- 如果满足则实现，否则提示先完成依赖

---

## 📋 任务信息获取机制

### 三层信息源（优先级从高到低）

1. **key-tasks.json**（13个关键任务的完整信息）
   - 优先查找此文件
   - 包含详细的实现指南、验收标准、代码示例引用
   - 涵盖：阶段入口任务 + 有完整代码参考的任务

2. **动态推断**（基于 task-template.md 规则）
   - 根据任务名称推断类型（Base类/实现类/脚本）
   - 根据阶段和类型推断文件路径
   - 根据类型生成标准验收标准

3. **TECH_SPEC.md**（技术规范参考）
   - 查找对应章节
   - 提取代码示例
   - 理解技术要求

### 任务信息查找流程

```
执行任务 X
  ↓
查找 key-tasks.json[X]
  ↓ 不存在
查看 task-map.json[X] 基础信息
  ↓
应用 task-template.md 规则推断
  ↓
查找 TECH_SPEC.md[spec_section] 参考
  ↓
组合生成完整任务信息
```

---

## 📊 完整任务列表（68 个任务）

**参考文件**：`.claude/skills/auto-dev/references/task-map.json`

### 阶段 A：工程骨架（3 个任务）
- A1: 初始化目录树 ⭐
- A2: pytest 测试框架
- A3: 配置加载 Settings

### 阶段 B：Libs 可插拔层（16 个任务）
- B1: BaseLLM ⭐
- B2-B8: LLM/Embedding/Vision/Splitter 实现
- B9: BaseVectorStore ⭐
- B10: MilvusStore ⭐✨
- B11-B16: Reranker/Evaluator/Factory

### 阶段 C：Ingestion Pipeline（15 个任务）
- C1: 数据类型定义 ⭐
- C2-C8: Loader/Transform/Encoder
- C9: SparseEncoder (BM25) ⭐✨
- C10-C15: Storage/Pipeline/脚本

### 阶段 D：Retrieval Pipeline（7 个任务）
- D1: QueryProcessor ⭐
- D2-D3: Retriever
- D4: RRF Fusion ⭐✨
- D5: HybridSearch ⭐✨
- D6-D7: Reranker/脚本

### 阶段 E-I：MCP/Trace/Dashboard/评估/验收
- E1, F1, G1, H1, I1 等阶段入口 ⭐

**图例**：
- ⭐ = key-tasks.json 中有详细信息
- ✨ = TECH_SPEC.md 中有完整代码参考

---

## 🔄 工作流程（遵循四个原则）

### 四个核心原则

1. **编码前思考** - 明确假设，呈现方案，询问不确定的地方
2. **简洁优先** - 只实现必需功能，避免过度工程
3. **精准修改** - 只创建必要文件，不改动无关代码
4. **目标驱动** - 定义验证标准，循环验证直到成功

### 执行步骤

#### 步骤 1：读取进度 + 获取任务信息
```markdown
## 检查进度

📁 读取进度文件：`.auto-dev-progress.json`

当前状态：
- 已完成：0/68 任务
- 当前阶段：A（工程骨架）
- 下一个任务：A1 - 初始化目录树

📋 获取任务详细信息：
1. 查找 key-tasks.json["A1"] ✅ 找到
2. 提取详细信息...
```

#### 步骤 2：任务分析（编码前思考）
```markdown
## 任务分析：A1 - 初始化目录树

### 从 key-tasks.json 提取信息：
- 描述：创建项目基本目录结构和空的入口文件
- 文件：src/__init__.py, config/settings.yaml, data/, logs/
- 预估：20行
- 验收标准：
  1. 目录结构符合 file-structure.md
  2. src/__init__.py 存在且可导入
  3. config/settings.yaml 是合法 YAML

### 我的理解和假设：
1. 创建基础目录结构
2. 创建空的 __init__.py 文件
3. 创建默认配置文件

### 不确定的地方：
- ❓ config/settings.yaml 需要什么默认配置？

### 建议方案：
**方案A（推荐）：** 最简实现
- 创建目录和空文件
- settings.yaml 包含基本结构（空配置）
- 优点：快速、简洁

---
**确认实现方案A？**
```

#### 步骤 3：定义成功标准
```markdown
## 成功标准

### 功能目标：
1. ✅ 创建目录结构 - 验证：目录存在
2. ✅ 创建 __init__.py - 验证：可导入
3. ✅ 创建 settings.yaml - 验证：YAML 合法

### 将创建的文件：
- src/__init__.py
- config/settings.yaml
- data/（目录）
- logs/（目录）

### 验证步骤：
1. 检查目录存在
2. Python 导入测试
3. YAML 解析测试
```

#### 步骤 4：实现 + 验证循环
生成代码 → 验证 → 如果失败则修复 → 重新验证

#### 步骤 5：更新进度
```markdown
## ✅ 任务完成：A1 - 初始化目录树

生成文件：
- src/__init__.py（0行，空文件）
- config/settings.yaml（10行）
- data/（目录）
- logs/（目录）

验证结果：
- ✅ 目录结构正确
- ✅ 可成功导入 src
- ✅ YAML 解析通过

更新进度：
- A1 标记为 completed
- 下一个任务：A2（pytest 测试框架）

---
**继续下一个任务？输入 /auto-dev 继续**
```

---

## 📝 关键文件说明

### references/key-tasks.json
包含 13 个关键任务的完整详细信息：
- 阶段入口任务（A1, B1, C1, D1, E1, F1, G1, H1, I1）
- 有完整代码参考的任务（B10, C9, D4, D5）

### references/task-template.md
动态推断规则和模板：
- 文件路径推断规则
- 验收标准模板
- 任务类型判断规则

### references/file-structure.md
项目文件结构规范

### references/task-map.json
全部 68 个任务的基础信息

---

## 🎯 关键要点

1. **三层信息源** - key-tasks.json → 动态推断 → TECH_SPEC.md
2. **自动化全流程** - 支持从 A1 到 I5 的完整开发
3. **进度可追踪** - 随时查看完成情况，新窗口可继续
4. **依赖自动检查** - 确保按正确顺序执行
5. **遵循四原则** - 编码前思考、简洁优先、精准修改、目标驱动
6. **循环验证** - 每个任务都要验证通过才标记完成

---

## 📝 注意事项

- TECH_SPEC.md 中的代码仅作**参考**，不是最终实现
- 实现基于最新的库 API，而非文档示例
- 优先考虑简洁性，避免过度工程
- 每个任务完成后更新进度文件
- 新窗口通过读取进度文件继续开发
- key-tasks.json 可以随时补充新的关键任务
