from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: dict


class DocumentLoader:
    """文档加载器，支持加载Markdown文件"""

    @staticmethod
    def load(directory: str) -> List[Document]:
        """
        加载指定目录下的所有.md文件

        Args:
            directory: 目录路径

        Returns:
            Document列表
        """
        documents = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        for file_path in sorted(dir_path.glob("*.md")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append(Document(
                content=content,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                }
            ))

        return documents
