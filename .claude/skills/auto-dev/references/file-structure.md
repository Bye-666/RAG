# 项目文件结构规范

基于 TECH_SPEC.md 4.2 节定义的目录结构。

## 标准目录树

```
RAG-MCP-SERVER/
│
├── config/                    # 配置文件目录
│   ├── settings.yaml          # 主配置文件
│   └── prompts/               # Prompt 模板目录
│
├── src/                       # 源代码主目录
│   ├── config/                # 配置加载模块
│   │   └── settings.py
│   │
│   ├── libs/                  # Libs 层（可插拔抽象层）
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   └── dashscope_llm.py
│   │   ├── embedding/
│   │   │   ├── base.py
│   │   │   └── dashscope_embedding.py
│   │   ├── vision_llm/
│   │   │   ├── base.py
│   │   │   └── dashscope_vision.py
│   │   ├── splitter/
│   │   │   ├── base.py
│   │   │   └── recursive_splitter.py
│   │   ├── vector_store/
│   │   │   ├── base.py
│   │   │   └── milvus_store.py
│   │   ├── reranker/
│   │   │   ├── base.py
│   │   │   └── cross_encoder.py
│   │   ├── evaluator/
│   │   │   ├── base.py
│   │   │   ├── ragas_evaluator.py
│   │   │   └── composite_evaluator.py
│   │   ├── factory.py
│   │   └── loader.py
│   │
│   ├── ingestion/             # Ingestion Pipeline
│   │   ├── types.py
│   │   ├── file_integrity.py
│   │   ├── loaders/
│   │   │   ├── base.py
│   │   │   └── pdf_loader.py
│   │   ├── transformers/
│   │   │   ├── chunk_refiner.py
│   │   │   ├── metadata_enricher.py
│   │   │   └── image_captioner.py
│   │   ├── embedders/
│   │   │   ├── dense_encoder.py
│   │   │   ├── sparse_encoder.py
│   │   │   └── batch_processor.py
│   │   ├── storage/
│   │   │   ├── vector_upserter.py
│   │   │   └── image_storage.py
│   │   └── pipeline.py
│   │
│   ├── core/                  # Core 层（核心业务逻辑）
│   │   └── query_engine/
│   │       ├── query_processor.py
│   │       ├── dense_retriever.py
│   │       ├── sparse_retriever.py
│   │       ├── rrf_fusion.py
│   │       ├── hybrid_search.py
│   │       └── reranker.py
│   │
│   ├── mcp_server/            # MCP Server 层
│   │   ├── server.py
│   │   ├── protocol_handler.py
│   │   └── tools/
│   │       ├── query_tool.py
│   │       ├── list_tool.py
│   │       └── summary_tool.py
│   │
│   └── observability/         # Observability 层
│       ├── trace/
│       │   ├── trace_context.py
│       │   ├── logger.py
│       │   └── trace_store.py
│       ├── dashboard/
│       │   ├── app.py
│       │   └── pages/
│       │       ├── overview.py
│       │       ├── data_browser.py
│       │       ├── ingestion_manager.py
│       │       ├── ingestion_traces.py
│       │       ├── query_traces.py
│       │       └── evaluation.py
│       └── evaluation/
│           └── golden_loader.py
│
├── data/                      # 数据目录
│   ├── documents/             # 原始文档存放
│   ├── images/                # 提取的图片存放
│   └── db/                    # 数据库与索引文件
│       ├── milvus.db          # Milvus Lite 数据文件
│       ├── bm25_encoder.pkl   # BM25 编码器
│       ├── ingestion_history.db
│       └── image_index.db
│
├── logs/                      # 日志目录
│   └── traces.jsonl           # 追踪日志（JSON Lines）
│
├── tests/                     # 测试目录
│   ├── unit/                  # 单元测试
│   │   ├── config/
│   │   ├── libs/
│   │   ├── ingestion/
│   │   ├── core/
│   │   ├── mcp_server/
│   │   └── observability/
│   ├── integration/           # 集成测试
│   │   ├── libs/
│   │   ├── ingestion/
│   │   └── core/
│   ├── e2e/                   # 端到端测试
│   │   ├── test_ingestion_e2e.py
│   │   ├── test_mcp_server_e2e.py
│   │   ├── test_mcp_client.py
│   │   ├── test_dashboard_smoke.py
│   │   └── test_full_pipeline.py
│   └── fixtures/              # 测试数据
│       └── golden_test_set.json
│
├── scripts/                   # 脚本目录
│   ├── ingest.py              # 数据摄取脚本
│   ├── query.py               # 查询测试脚本
│   ├── evaluate.py            # 评估运行脚本
│   └── start_dashboard.py     # Dashboard 启动脚本
│
├── .claude/                   # Claude 配置
│   └── skills/
│       └── auto-dev/
│
├── .auto-dev-progress.json    # 自动开发进度文件
├── main.py                    # MCP Server 启动入口
├── pytest.ini                 # pytest 配置
├── pyproject.toml
├── requirements.txt
├── TECH_SPEC.md
└── README.md
```

## 文件命名规范

### Python 模块
- **snake_case**：`milvus_store.py`, `query_processor.py`
- 类名：PascalCase - `MilvusStore`, `QueryProcessor`
- 函数/方法：snake_case - `search_dense()`, `encode_batch()`

### 测试文件
- 前缀 `test_`：`test_milvus_store.py`
- 与源文件对应：`src/libs/vector_store/milvus_store.py` → `tests/unit/libs/vector_store/test_milvus_store.py`

### 配置文件
- YAML：`settings.yaml`
- JSON：`task-map.json`, `progress.json`

## 导入路径规范

```python
# 绝对导入（推荐）
from src.libs.vector_store.milvus_store import MilvusStore
from src.ingestion.types import Document, Chunk

# 相对导入（同级或子级模块）
from .base import BaseVectorStore
from ..embedders.dense_encoder import DenseEncoder
```

## __init__.py 使用

- **必需**：每个 Python 包目录都应有 `__init__.py`
- **可为空**：快速原型阶段可以是空文件
- **暴露 API**：成熟后用于暴露公共接口

```python
# src/libs/vector_store/__init__.py
from .base import BaseVectorStore
from .milvus_store import MilvusStore

__all__ = ["BaseVectorStore", "MilvusStore"]
```

## 测试目录镜像规则

测试目录结构完全镜像源代码目录：

```
src/libs/vector_store/milvus_store.py
  ↓
tests/unit/libs/vector_store/test_milvus_store.py
```
