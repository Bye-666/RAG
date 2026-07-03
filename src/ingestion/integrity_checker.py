"""文件完整性检查器

使用 SHA256 哈希检测文件是否发生变化，避免重复处理。
"""
import hashlib
import json
from pathlib import Path
from typing import Optional


class FileIntegrityChecker:
    """文件完整性检查器

    使用 SHA256 哈希值跟踪文件变化。
    """

    def __init__(self, index_file: str = "./data/db/file_integrity.json"):
        """初始化检查器

        Args:
            index_file: 索引文件路径
        """
        self.index_file = Path(index_file)
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """加载索引文件"""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """保存索引文件"""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def compute_hash(self, file_path: str) -> str:
        """计算文件 SHA256 哈希

        Args:
            file_path: 文件路径

        Returns:
            SHA256 哈希值
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_changed(self, file_path: str) -> bool:
        """检查文件是否发生变化

        Args:
            file_path: 文件路径

        Returns:
            True 表示文件已变化或首次处理
        """
        current_hash = self.compute_hash(file_path)
        stored_hash = self.index.get(file_path)

        return current_hash != stored_hash

    def update(self, file_path: str) -> None:
        """更新文件哈希记录

        Args:
            file_path: 文件路径
        """
        current_hash = self.compute_hash(file_path)
        self.index[file_path] = current_hash
        self._save_index()

    def remove(self, file_path: str) -> None:
        """移除文件记录

        Args:
            file_path: 文件路径
        """
        if file_path in self.index:
            del self.index[file_path]
            self._save_index()

    def get_hash(self, file_path: str) -> Optional[str]:
        """获取文件已存储的哈希值

        Args:
            file_path: 文件路径

        Returns:
            哈希值，如果不存在则返回 None
        """
        return self.index.get(file_path)
