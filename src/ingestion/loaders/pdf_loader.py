"""PDF Loader 实现

使用 MarkItDown 将 PDF 转换为 Markdown。
"""
import hashlib
from pathlib import Path

from .base import BaseLoader
from ..types import Document


class PDFLoader(BaseLoader):
    """PDF 加载器

    使用 MarkItDown 库解析 PDF 为 Markdown 格式。
    """

    def __init__(self):
        """初始化 PDF Loader"""
        try:
            from markitdown import MarkItDown
            self.converter = MarkItDown()
        except ImportError:
            raise ImportError("markitdown 包未安装。请运行：pip install markitdown")

    def load(self, file_path: str) -> Document:
        """加载 PDF 文件

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象
        """
        path = Path(file_path)

        # 转换为 Markdown
        result = self.converter.convert(str(path))
        text = result.text_content

        # 生成文档 ID
        doc_id = hashlib.sha256(str(path).encode()).hexdigest()[:16]

        # 创建 Document
        return Document(
            id=doc_id,
            text=text,
            source=str(path),
            metadata={
                "filename": path.name,
                "file_type": "pdf"
            }
        )
