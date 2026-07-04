"""Clear all documents and related records.

This script will:
1. Delete Milvus vector database
2. Delete BM25 encoder
3. Clear trace records
4. Clear uploaded documents (optional)

Usage:
    python scripts/clear_all_data.py
"""

import shutil
from pathlib import Path


def clear_all_data():
    """Clear all data and records."""
    print("清空所有文档和记录")
    print("=" * 50)

    items_to_clear = []

    # 1. Milvus database
    milvus_db = Path("data/db/milvus.db")
    if milvus_db.exists():
        items_to_clear.append(("Milvus 向量数据库", milvus_db))

    # 2. BM25 encoder
    bm25_encoder = Path("data/db/bm25_encoder.pkl")
    if bm25_encoder.exists():
        items_to_clear.append(("BM25 编码器", bm25_encoder))

    # 3. Trace records
    trace_file = Path("data/traces/traces.json")
    if trace_file.exists():
        items_to_clear.append(("追踪记录", trace_file))

    # 4. Documents directory (optional - commented out by default)
    # documents_dir = Path("data/documents")
    # if documents_dir.exists() and any(documents_dir.iterdir()):
    #     items_to_clear.append(("上传的文档", documents_dir))

    if not items_to_clear:
        print("\n没有需要清空的数据")
        return

    # Show what will be deleted
    print("\n将删除以下数据：")
    for name, path in items_to_clear:
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            print(f"  - {name}: {path} ({size / 1024:.1f} KB)")
        else:
            size = path.stat().st_size
            print(f"  - {name}: {path} ({size / 1024:.1f} KB)")

    # Confirm
    print("\n警告：此操作不可恢复！")
    confirm = input("确认清空所有数据？(yes/no): ").strip().lower()

    if confirm != "yes":
        print("\n已取消")
        return

    # Delete
    print("\n开始清空...")
    for name, path in items_to_clear:
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"✓ 已删除 {name}")
            else:
                path.unlink()
                print(f"✓ 已删除 {name}")
        except Exception as e:
            print(f"✗ 删除 {name} 失败: {e}")

    print("\n" + "=" * 50)
    print("清空完成！")
    print("\n下一步：")
    print("1. 重启 Streamlit 应用")
    print("2. 重新上传并摄取文档")
    print("3. 新的 BM25 编码器将使用中文分词训练")


if __name__ == "__main__":
    try:
        clear_all_data()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
