# 任务信息模板

用于动态推断任务详细信息的模板。

## 任务信息结构

```json
{
  "name": "任务名称",
  "description": "详细描述：做什么，为什么",
  "src_files": ["源文件路径列表"],
  "test_files": ["测试文件路径列表"],
  "estimated_lines": 100,
  "spec_section": "TECH_SPEC.md 对应章节",
  "has_complete_code": false,
  "success_criteria": [
    "验收标准1",
    "验收标准2"
  ],
  "key_methods": ["方法1", "方法2"],
  "dependencies_detail": {
    "external": ["外部依赖包"],
    "internal": ["项目内部依赖"]
  },
  "implementation_notes": [
    "实现要点1",
    "实现要点2"
  ]
}
```

## 字段说明

### 必需字段

- **name**: 任务简短名称
- **description**: 详细描述，说明做什么
- **src_files**: 将创建的源代码文件列表
- **success_criteria**: 验收标准列表

### 可选字段

- **test_files**: 测试文件列表（如果有）
- **estimated_lines**: 预估代码行数
- **spec_section**: TECH_SPEC.md 对应章节
- **has_complete_code**: 文档中是否有完整代码参考
- **key_methods**: 核心方法/函数名称
- **key_types**: 核心类型定义（对于类型定义任务）
- **key_functions**: 核心函数（对于函数式模块）
- **dependencies_detail**: 依赖详情
  - external: 外部库依赖
  - internal: 项目内部模块依赖
- **implementation_notes**: 实现要点和注意事项
- **notes**: 其他备注
- **is_modification**: 是否是修改现有文件
- **is_documentation**: 是否是文档任务

## 文件路径推断规则

### 源文件路径规则

根据任务类型和阶段自动推断：

**阶段 A（工程骨架）**：
- 配置相关 → `src/config/`
- 根目录文件 → 项目根

**阶段 B（Libs层）**：
- LLM → `src/libs/llm/`
- Embedding → `src/libs/embedding/`
- VisionLLM → `src/libs/vision_llm/`
- Splitter → `src/libs/splitter/`
- VectorStore → `src/libs/vector_store/`
- Reranker → `src/libs/reranker/`
- Evaluator → `src/libs/evaluator/`
- Factory → `src/libs/`

**阶段 C（Ingestion）**：
- 类型定义 → `src/ingestion/types.py`
- Loader → `src/ingestion/loaders/`
- Transform → `src/ingestion/transformers/`
- Embedder → `src/ingestion/embedders/`
- Storage → `src/ingestion/storage/`
- Pipeline → `src/ingestion/pipeline.py`

**阶段 D（Retrieval）**：
- 所有组件 → `src/core/query_engine/`

**阶段 E（MCP Server）**：
- Server → `src/mcp_server/server.py`
- Handler → `src/mcp_server/protocol_handler.py`
- Tools → `src/mcp_server/tools/`

**阶段 F（Trace）**：
- 所有组件 → `src/observability/trace/`

**阶段 G（Dashboard）**：
- App → `src/observability/dashboard/app.py`
- Pages → `src/observability/dashboard/pages/`

**阶段 H（Evaluator）**：
- Evaluator → `src/libs/evaluator/`
- Golden → `tests/fixtures/` 和 `src/observability/evaluation/`

**阶段 I（E2E）**：
- 所有测试 → `tests/e2e/`
- 文档 → 项目根

### 测试文件路径规则

- 单元测试：`tests/unit/` + 源文件相对路径
  - `src/libs/vector_store/milvus_store.py` 
  - → `tests/unit/libs/vector_store/test_milvus_store.py`
  
- 集成测试：`tests/integration/` + 模块路径
  
- E2E 测试：`tests/e2e/test_<功能>_e2e.py`

## 验收标准模板

### 抽象接口（Base类）
```
- 定义 Base<Name> 抽象类
- 包含核心方法：method1(), method2()
- 定义输入输出类型
- 使用 ABC 抽象基类
```

### 具体实现
```
- 继承 Base<Name>
- 实现所有抽象方法
- 单元测试覆盖核心功能
- 错误处理完整
```

### Pipeline/编排类
```
- 整合所有子组件
- 实现完整流程
- 集成测试验证端到端
- 支持进度回调（如有）
```

### 脚本入口
```
- 可执行脚本
- 参数解析完整
- 错误提示友好
- 有使用说明
```

## 动态推断流程

当执行一个任务时：

1. **优先查找 key-tasks.json**
   - 如果存在 → 直接使用详细信息
   
2. **否则，从 task-map.json 基础信息 + 规则推断**
   - 根据任务名称推断类型（Base类/实现类/脚本等）
   - 根据阶段和类型推断文件路径
   - 根据类型生成标准验收标准
   
3. **从 TECH_SPEC.md 查找参考**
   - 如果有 spec_section → 读取对应章节
   - 查找是否有代码示例

## 示例

### 示例1：推断 B3 (BaseEmbedding)

```json
{
  "name": "BaseEmbedding",
  "description": "定义 Embedding 的抽象基类",
  "src_files": ["src/libs/embedding/base.py"],
  "test_files": [],
  "estimated_lines": 40,
  "success_criteria": [
    "定义 BaseEmbedding 抽象类",
    "包含核心方法：encode(), encode_batch()",
    "定义输入输出类型"
  ]
}
```

### 示例2：推断 C14 (ingest.py 脚本)

```json
{
  "name": "ingest.py 脚本",
  "description": "数据摄取脚本入口",
  "src_files": ["scripts/ingest.py"],
  "test_files": [],
  "estimated_lines": 100,
  "success_criteria": [
    "可执行脚本",
    "参数解析（文件路径、配置等）",
    "调用 Pipeline 执行摄取",
    "错误提示友好"
  ]
}
```
