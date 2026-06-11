from dataclasses import dataclass
from typing import List
import re
from .document_loader import Document


@dataclass
class Chunk:
    """文档块数据结构"""
    text: str
    source: str
    metadata: dict


class Chunker:
    """
    文档切分器，支持按Markdown标题层级切分

    切分策略：
    1. 优先按 ## 二级标题切分
    2. 如果二级标题下内容过长，再按 ### 三级标题切分
    3. 过小的 chunk（< min_chunk_size）会与相邻 chunk 合并
    4. 确保 chunk 不以下一个标题结尾
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 80, min_chunk_size: int = 120):
        """
        初始化切分器

        Args:
            chunk_size: 最大块大小（字符数），默认800
            overlap: 相邻块重叠字符数，默认80
            min_chunk_size: 最小块大小，过小的块会被合并，默认120

        Raises:
            ValueError: 参数不合法时抛出
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if overlap < 0:
            raise ValueError("overlap 不能为负数")
        if overlap >= chunk_size:
            raise ValueError("overlap 必须小于 chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size 不能为负数")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def split(self, documents: List[Document]) -> List[Chunk]:
        """
        切分文档列表

        Args:
            documents: Document列表

        Returns:
            Chunk列表
        """
        chunks = []

        for doc in documents:
            doc_chunks = self._split_document(doc)
            chunks.extend(doc_chunks)

        return chunks

    def _split_document(self, document: Document) -> List[Chunk]:
        """切分单个文档"""
        content = document.content
        source = document.metadata.get("source", "")
        document_title = self._extract_document_title(content)

        # 步骤1: 按 ## 二级标题切分
        level2_sections = self._split_by_level2_header(content)

        # 步骤2: 对每个二级标题区域，检查是否需要进一步按 ### 切分
        all_sections = []
        for section_text, parent_header, current_header in level2_sections:
            if len(section_text) > self.chunk_size:
                # 过长，按三级标题进一步切分
                level3_sections = self._split_by_level3_header(section_text, parent_header)
                all_sections.extend(level3_sections)
            else:
                all_sections.append((section_text, parent_header, current_header))

        # 步骤3: 清理每个 section 的边界（移除结尾的下一个标题）
        cleaned_sections = []
        for section_text, parent_header, current_header in all_sections:
            cleaned_text = self._clean_section_boundary(section_text)
            if cleaned_text.strip():
                cleaned_sections.append((cleaned_text, parent_header, current_header))

        # 步骤4: 合并过小的 section
        merged_sections = self._merge_small_sections(cleaned_sections)

        # 步骤5: 生成 Chunk
        chunks = []
        for section_text, parent_header, current_header in merged_sections:
            if not section_text.strip():
                continue

            metadata = {
                **document.metadata,
                "document_title": document_title,
                "parent_header": parent_header,
                "current_header": current_header,
                "header": current_header,
            }

            # 如果仍然过长，按字符数切分
            if len(section_text) > self.chunk_size:
                sub_chunks = self._split_by_size(section_text)
                for sub_text in sub_chunks:
                    chunks.append(Chunk(
                        text=sub_text.strip(),
                        source=source,
                        metadata=metadata.copy(),
                    ))
            else:
                chunks.append(Chunk(
                    text=section_text.strip(),
                    source=source,
                    metadata=metadata.copy(),
                ))

        return chunks

    def _extract_document_title(self, content: str) -> str:
        """提取 Markdown 一级标题。"""
        for line in content.splitlines():
            if re.match(r'^#\s+', line):
                return line.strip()
        return ""

    def _split_by_level2_header(self, content: str) -> List[tuple[str, str, str]]:
        """
        按 ## 二级标题切分文档

        Args:
            content: 文档内容

        Returns:
            (section_text, parent_header, current_header) 元组列表
        """
        lines = content.split('\n')
        sections = []
        current_header = ""
        current_text = []

        for line in lines:
            # 匹配 ## 二级标题（不包括 ###）
            if re.match(r'^##\s+', line):
                # 保存之前的 section
                if current_text:
                    sections.append(('\n'.join(current_text), current_header, current_header))
                current_header = line
                current_text = [line]
            else:
                current_text.append(line)

        # 保存最后一个 section
        if current_text:
            sections.append(('\n'.join(current_text), current_header, current_header))

        return sections if sections else [(content, "", "")]

    def _split_by_level3_header(self, section_text: str, parent_header: str) -> List[tuple[str, str, str]]:
        """
        在二级标题区域内，按 ### 三级标题进一步切分

        Args:
            section_text: 二级标题区域的文本
            parent_header: 父级二级标题

        Returns:
            (section_text, header) 元组列表
        """
        lines = section_text.split('\n')
        sections = []
        current_header = parent_header
        current_text = []

        for line in lines:
            # 匹配 ### 三级标题
            if re.match(r'^###\s+', line):
                # 保存之前的 section
                if current_text:
                    sections.append(('\n'.join(current_text), parent_header, current_header))
                current_header = line
                current_text = [line]
            else:
                current_text.append(line)

        # 保存最后一个 section
        if current_text:
            sections.append(('\n'.join(current_text), parent_header, current_header))

        return sections if sections else [(section_text, parent_header, parent_header)]

    def _clean_section_boundary(self, section_text: str) -> str:
        """
        清理 section 边界，移除结尾的下一个标题

        Args:
            section_text: 原始 section 文本

        Returns:
            清理后的文本
        """
        lines = section_text.rstrip().split('\n')

        # 从末尾移除独立的标题行（没有后续内容的标题）
        while lines:
            last_line = lines[-1].strip()
            # 如果最后一行是标题，移除它
            if re.match(r'^#{1,3}\s+', last_line):
                lines.pop()
            else:
                break

        return '\n'.join(lines)

    def _merge_small_sections(self, sections: List[tuple[str, str, str]]) -> List[tuple[str, str, str]]:
        """
        合并过小的 section 到相邻 section

        Args:
            sections: (section_text, header) 元组列表

        Returns:
            合并后的 sections 列表
        """
        if not sections:
            return []

        merged = []
        i = 0

        while i < len(sections):
            current_text, current_parent, current_header = sections[i]
            current_len = len(current_text.strip())

            # 如果当前 section 过小
            if current_len < self.min_chunk_size:
                # 优先合并到前一个 section
                if merged:
                    prev_text, prev_parent, prev_header = merged[-1]
                    combined = prev_text + "\n\n" + current_text
                    # 只在同一父标题内合并，避免小节被并入其他主题
                    if (
                        prev_parent == current_parent
                        and prev_header == current_header
                        and len(combined) <= self.chunk_size * 1.5
                    ):
                        merged[-1] = (combined, prev_parent, prev_header)
                        i += 1
                        continue

                # 否则合并到下一个 section
                if i + 1 < len(sections):
                    next_text, next_parent, next_header = sections[i + 1]
                    if (
                        next_parent == current_parent
                        and next_header == current_header
                    ):
                        combined = current_text + "\n\n" + next_text
                        sections[i + 1] = (combined, current_parent, current_header)
                        i += 1
                        continue

            # 当前 section 大小合适，直接添加
            merged.append((current_text, current_parent, current_header))
            i += 1

        return merged

    def _split_by_size(self, text: str) -> List[str]:
        """
        按字符数切分长文本

        Args:
            text: 待切分文本

        Returns:
            切分后的文本块列表
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 尝试在句子边界切分
            if end < len(text):
                # 在 chunk 末尾附近找句子结束符
                search_start = max(start + self.chunk_size - 100, start)
                search_text = text[search_start:end + 50]

                # 查找句子结束位置
                sentence_end = -1
                for marker in ['。\n', '。\n\n', '。\n', '。\n']:
                    pos = search_text.rfind(marker)
                    if pos > 0:
                        sentence_end = search_start + pos + len(marker) - 1
                        break

                if sentence_end > start and sentence_end <= end + 50:
                    end = sentence_end

            # 确保不超出文本长度
            if end >= len(text):
                chunks.append(text[start:])
                break

            chunk = text[start:end]
            chunks.append(chunk)

            # 计算下一个起始位置（考虑 overlap）
            next_start = end - self.overlap

            # 确保至少前进
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks
