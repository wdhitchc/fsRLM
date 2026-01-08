"""
Chunking utilities for ccRLM.

This module provides helpers for splitting large text into chunks
suitable for recursive LLM processing. It handles:
- Line-based chunking
- Byte-based chunking
- Token-aware chunking (approximate)
- Chunk manifest generation
- Index building for prompt files

Usage in scripts:
    from tools.chunking import Chunker

    chunker = Chunker()
    chunks = chunker.chunk_file("input/prompt.md", max_tokens=2000)

    for chunk in chunks:
        result = call_claude(f"Summarize: {chunk.content}")
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    content: str
    index: int
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    token_estimate: int
    source_file: str
    chunk_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkManifest:
    """Manifest describing how a file was chunked."""
    source_file: str
    total_lines: int
    total_bytes: int
    total_tokens_estimate: int
    chunk_count: int
    chunk_method: str
    chunks: list[dict]
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ChunkManifest":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class Chunker:
    """
    Utility for chunking large text files.

    Supports multiple chunking strategies:
    - by_lines: Fixed number of lines per chunk
    - by_tokens: Approximate token count per chunk
    - by_bytes: Fixed byte size per chunk
    - by_paragraphs: Split on blank lines
    """

    def __init__(self, workspace_root: Optional[Path] = None):
        """
        Initialize chunker.

        Args:
            workspace_root: Path to workspace. If None, uses cwd.
        """
        self.workspace = Path(workspace_root or os.getcwd())
        self.indexes_dir = self.workspace / "cache" / "indexes"
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _generate_chunk_id(self, source: str, index: int, content: str) -> str:
        """Generate a unique chunk ID."""
        key = f"{source}:{index}:{hashlib.md5(content.encode()).hexdigest()[:8]}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def chunk_by_lines(
        self,
        text: str,
        lines_per_chunk: int = 100,
        overlap_lines: int = 5,
        source_file: str = "unknown",
    ) -> list[Chunk]:
        """
        Split text into chunks by line count.

        Args:
            text: Text to chunk
            lines_per_chunk: Lines per chunk
            overlap_lines: Lines to overlap between chunks
            source_file: Source filename for metadata

        Returns:
            List of Chunk objects
        """
        lines = text.split("\n")
        chunks = []

        start_line = 0
        chunk_index = 0
        current_byte = 0

        while start_line < len(lines):
            end_line = min(start_line + lines_per_chunk, len(lines))
            chunk_lines = lines[start_line:end_line]
            content = "\n".join(chunk_lines)

            chunk = Chunk(
                content=content,
                index=chunk_index,
                start_line=start_line + 1,  # 1-indexed
                end_line=end_line,
                start_byte=current_byte,
                end_byte=current_byte + len(content),
                token_estimate=self._estimate_tokens(content),
                source_file=source_file,
                chunk_id=self._generate_chunk_id(source_file, chunk_index, content),
            )
            chunks.append(chunk)

            current_byte += len(content) + 1  # +1 for newline
            start_line = end_line - overlap_lines if overlap_lines > 0 else end_line
            chunk_index += 1

            # Prevent infinite loop
            if start_line >= len(lines) - overlap_lines:
                break

        return chunks

    def chunk_by_tokens(
        self,
        text: str,
        max_tokens: int = 2000,
        overlap_tokens: int = 100,
        source_file: str = "unknown",
    ) -> list[Chunk]:
        """
        Split text into chunks by approximate token count.

        Args:
            text: Text to chunk
            max_tokens: Max tokens per chunk
            overlap_tokens: Tokens to overlap
            source_file: Source filename

        Returns:
            List of Chunk objects
        """
        # Convert token targets to character targets (4 chars per token)
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4

        lines = text.split("\n")
        chunks = []

        current_chunk_lines = []
        current_chunk_chars = 0
        chunk_index = 0
        start_line = 0
        current_byte = 0

        for i, line in enumerate(lines):
            line_chars = len(line) + 1  # +1 for newline

            if current_chunk_chars + line_chars > max_chars and current_chunk_lines:
                # Emit chunk
                content = "\n".join(current_chunk_lines)
                chunk = Chunk(
                    content=content,
                    index=chunk_index,
                    start_line=start_line + 1,
                    end_line=start_line + len(current_chunk_lines),
                    start_byte=current_byte,
                    end_byte=current_byte + len(content),
                    token_estimate=self._estimate_tokens(content),
                    source_file=source_file,
                    chunk_id=self._generate_chunk_id(source_file, chunk_index, content),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_line_count = 0
                overlap_char_count = 0
                overlap_lines = []

                for prev_line in reversed(current_chunk_lines):
                    if overlap_char_count + len(prev_line) > overlap_chars:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_char_count += len(prev_line) + 1
                    overlap_line_count += 1

                current_byte += len(content) - overlap_char_count + 1
                start_line = start_line + len(current_chunk_lines) - overlap_line_count
                current_chunk_lines = overlap_lines
                current_chunk_chars = overlap_char_count
                chunk_index += 1

            current_chunk_lines.append(line)
            current_chunk_chars += line_chars

        # Emit final chunk
        if current_chunk_lines:
            content = "\n".join(current_chunk_lines)
            chunk = Chunk(
                content=content,
                index=chunk_index,
                start_line=start_line + 1,
                end_line=start_line + len(current_chunk_lines),
                start_byte=current_byte,
                end_byte=current_byte + len(content),
                token_estimate=self._estimate_tokens(content),
                source_file=source_file,
                chunk_id=self._generate_chunk_id(source_file, chunk_index, content),
            )
            chunks.append(chunk)

        return chunks

    def chunk_by_paragraphs(
        self,
        text: str,
        max_paragraphs: int = 5,
        source_file: str = "unknown",
    ) -> list[Chunk]:
        """
        Split text by paragraphs (blank line separated).

        Args:
            text: Text to chunk
            max_paragraphs: Max paragraphs per chunk
            source_file: Source filename

        Returns:
            List of Chunk objects
        """
        # Split on double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []

        chunk_index = 0
        current_byte = 0
        current_line = 1

        for i in range(0, len(paragraphs), max_paragraphs):
            para_group = paragraphs[i:i + max_paragraphs]
            content = "\n\n".join(para_group)

            line_count = content.count("\n") + 1

            chunk = Chunk(
                content=content,
                index=chunk_index,
                start_line=current_line,
                end_line=current_line + line_count - 1,
                start_byte=current_byte,
                end_byte=current_byte + len(content),
                token_estimate=self._estimate_tokens(content),
                source_file=source_file,
                chunk_id=self._generate_chunk_id(source_file, chunk_index, content),
            )
            chunks.append(chunk)

            current_byte += len(content) + 2  # +2 for paragraph separator
            current_line += line_count + 1
            chunk_index += 1

        return chunks

    def chunk_file(
        self,
        file_path: str,
        method: str = "tokens",
        max_tokens: int = 2000,
        lines_per_chunk: int = 100,
        overlap: int = 100,
        save_manifest: bool = True,
    ) -> list[Chunk]:
        """
        Chunk a file and optionally save a manifest.

        Args:
            file_path: Path to file (relative to workspace)
            method: "tokens", "lines", or "paragraphs"
            max_tokens: For token-based chunking
            lines_per_chunk: For line-based chunking
            overlap: Overlap amount (tokens or lines depending on method)
            save_manifest: Whether to save manifest to cache/indexes/

        Returns:
            List of Chunk objects
        """
        full_path = self.workspace / file_path
        text = full_path.read_text()

        if method == "tokens":
            chunks = self.chunk_by_tokens(
                text,
                max_tokens=max_tokens,
                overlap_tokens=overlap,
                source_file=file_path,
            )
        elif method == "lines":
            chunks = self.chunk_by_lines(
                text,
                lines_per_chunk=lines_per_chunk,
                overlap_lines=overlap if overlap < lines_per_chunk else 5,
                source_file=file_path,
            )
        elif method == "paragraphs":
            chunks = self.chunk_by_paragraphs(
                text,
                max_paragraphs=5,
                source_file=file_path,
            )
        else:
            raise ValueError(f"Unknown chunking method: {method}")

        # Create and save manifest
        if save_manifest:
            manifest = ChunkManifest(
                source_file=file_path,
                total_lines=text.count("\n") + 1,
                total_bytes=len(text),
                total_tokens_estimate=self._estimate_tokens(text),
                chunk_count=len(chunks),
                chunk_method=method,
                chunks=[c.to_dict() for c in chunks],
                created_at=datetime.utcnow().isoformat(),
            )

            manifest_name = Path(file_path).stem + "_chunks.json"
            manifest.save(self.indexes_dir / manifest_name)

        return chunks

    def build_index(
        self,
        file_path: str,
        index_type: str = "headings",
    ) -> dict:
        """
        Build an index of a file for quick navigation.

        Args:
            file_path: Path to file
            index_type: "headings" (markdown) or "structure"

        Returns:
            Index dictionary
        """
        full_path = self.workspace / file_path
        text = full_path.read_text()
        lines = text.split("\n")

        index = {
            "source_file": file_path,
            "total_lines": len(lines),
            "created_at": datetime.utcnow().isoformat(),
            "entries": [],
        }

        if index_type == "headings":
            # Find markdown headings
            for i, line in enumerate(lines):
                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    title = line.lstrip("#").strip()
                    index["entries"].append({
                        "type": "heading",
                        "level": level,
                        "title": title,
                        "line": i + 1,
                    })

        # Save index
        index_name = Path(file_path).stem + f"_{index_type}_index.json"
        with open(self.indexes_dir / index_name, "w") as f:
            json.dump(index, f, indent=2)

        return index

    def get_lines(self, file_path: str, start: int, end: int) -> str:
        """
        Get specific lines from a file.

        Args:
            file_path: Path to file
            start: Start line (1-indexed, inclusive)
            end: End line (1-indexed, inclusive)

        Returns:
            Text content of those lines
        """
        full_path = self.workspace / file_path
        lines = full_path.read_text().split("\n")
        return "\n".join(lines[start - 1:end])


# Convenience functions for simple usage
def chunk_prompt(max_tokens: int = 2000) -> list[Chunk]:
    """
    Chunk the main prompt file (input/prompt.md).

    Example:
        from tools.chunking import chunk_prompt
        for chunk in chunk_prompt(2000):
            process(chunk.content)
    """
    chunker = Chunker()
    return chunker.chunk_file("input/prompt.md", method="tokens", max_tokens=max_tokens)


def get_prompt_lines(start: int, end: int) -> str:
    """
    Get specific lines from the prompt file.

    Example:
        from tools.chunking import get_prompt_lines
        section = get_prompt_lines(100, 150)
    """
    chunker = Chunker()
    return chunker.get_lines("input/prompt.md", start, end)
