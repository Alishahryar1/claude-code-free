"""Query classifier for detecting request types from user messages."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from enum import Enum

class QueryType(str, Enum):
    """Types of queries that can be classified."""

    CODE = "code"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    VISION = "vision"
    GENERAL = "general"
    REASONING = "reasoning"
    MATH = "math"
    WRITING = "writing"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Result of query classification."""

    query_type: QueryType
    confidence: float
    reasoning: str


class QueryClassifier:
    """Classify user queries to determine the best model to use."""

    def __init__(self) -> None:
        self._code_keywords = {
            "code",
            "function",
            "class",
            "method",
            "variable",
            "import",
            "export",
            "debug",
            "fix",
            "bug",
            "error",
            "syntax",
            "compile",
            "build",
            "deploy",
            "test",
            "refactor",
            "optimize",
            "algorithm",
            "data structure",
            "api",
            "endpoint",
            "database",
            "sql",
            "query",
            "schema",
            "migration",
            "git",
            "commit",
            "branch",
            "merge",
            "pull request",
            "pr",
            "issue",
            "feature",
            "implementation",
            "integration",
            "unit test",
            "e2e",
            "end-to-end",
            "mock",
            "stub",
            "dependency",
            "package",
            "library",
            "framework",
            "react",
            "vue",
            "angular",
            "node",
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "c++",
            "c#",
            "php",
            "ruby",
            "swift",
            "kotlin",
            "dart",
            "scala",
            "haskell",
            "elixir",
            "erlang",
            "clojure",
            "f#",
            "racket",
            "scheme",
            "lisp",
            "prolog",
            "sql",
            "graphql",
            "rest",
            "grpc",
            "websocket",
            "http",
            "https",
            "json",
            "xml",
            "yaml",
            "toml",
            "ini",
            "csv",
            "docker",
            "kubernetes",
            "k8s",
            "helm",
            "terraform",
            "ansible",
            "chef",
            "puppet",
            "ci/cd",
            "jenkins",
            "github actions",
            "gitlab ci",
            "circleci",
            "travis",
            "aws",
            "azure",
            "gcp",
            "lambda",
            "function",
            "serverless",
            "microservice",
            "monolith",
            "architecture",
            "design pattern",
            "solid",
            "dry",
            "kiss",
            "yagni",
            "clean code",
            "code review",
            "pull request",
            "merge request",
            "code smell",
            "technical debt",
            "legacy",
            "spaghetti",
            "god object",
            "singleton",
            "factory",
            "observer",
            "strategy",
            "decorator",
            "adapter",
            "facade",
            "proxy",
            "composite",
            "iterator",
            "command",
            "chain of responsibility",
            "mediator",
            "memento",
            "state",
            "template method",
            "visitor",
            "builder",
            "prototype",
        }

        self._vision_keywords = {
            "image",
            "picture",
            "photo",
            "screenshot",
            "diagram",
            "chart",
            "graph",
            "plot",
            "visual",
            "vision",
            "see",
            "look",
            "describe",
            "analyze image",
            "caption",
            "ocr",
            "text from image",
            "read image",
            "extract from image",
            "what's in this",
            "what do you see",
            "show me",
            "draw",
            "paint",
            "sketch",
            "illustration",
            "graphic",
            "design",
            "logo",
            "icon",
            "emoji",
            "meme",
            "gif",
            "video",
            "frame",
            "scene",
            "object detection",
            "face recognition",
            "segmentation",
            "classification",
            "multimodal",
            "vl",
            "vision-language",
        }

        self._summarization_keywords = {
            "summarize",
            "summary",
            "summarization",
            "condense",
            "shorten",
            "brief",
            "overview",
            "recap",
            "tldr",
            "too long",
            "key points",
            "main idea",
            "essence",
            "gist",
            "abstract",
            "executive summary",
            "bullet points",
            "outline",
            "highlights",
            "takeaways",
            "in a nutshell",
            "in short",
            "basically",
            "bottom line",
            "to sum up",
            "in conclusion",
            "wrap up",
            "simplify",
            "explain simply",
            "for dummies",
            "explain like i'm 5",
            "eli5",
            "quick summary",
            "one sentence",
            "one paragraph",
            "briefly explain",
            "give me the gist",
            "what's the point",
            "main takeaway",
        }

        self._reasoning_keywords = {
            "reason",
            "think",
            "explain why",
            "why does",
            "how does",
            "how would",
            "what if",
            "consider",
            "analyze",
            "evaluate",
            "assess",
            "critique",
            "compare",
            "contrast",
            "pros and cons",
            "advantages",
            "disadvantages",
            "implications",
            "consequences",
            "logic",
            "deduction",
            "inference",
            "argument",
            "premise",
            "conclusion",
            "fallacy",
            "paradox",
            "dilemma",
            "trade-off",
            "decision",
            "judgment",
            "recommendation",
            "advice",
            "opinion",
            "perspective",
            "viewpoint",
            "standpoint",
            "angle",
            "approach",
            "methodology",
            "framework",
            "theory",
            "hypothesis",
            "thesis",
            "antithesis",
            "synthesis",
            "step by step",
            "walk through",
            "break down",
            "deconstruct",
            "unpack",
            "elaborate",
            "expand",
            "go deeper",
            "dig into",
            "explore",
            "investigate",
            "examine",
            "scrutinize",
            "delve",
            "ponder",
            "reflect",
            "contemplate",
            "meditate",
            "ruminate",
        }

        self._math_keywords = {
            "calculate",
            "compute",
            "solve",
            "equation",
            "formula",
            "math",
            "mathematics",
            "algebra",
            "calculus",
            "geometry",
            "trigonometry",
            "statistics",
            "probability",
            "arithmetic",
            "addition",
            "subtraction",
            "multiplication",
            "division",
            "fraction",
            "decimal",
            "percentage",
            "ratio",
            "proportion",
            "graph",
            "function",
            "derivative",
            "integral",
            "limit",
            "matrix",
            "vector",
            "tensor",
            "complex number",
            "imaginary",
            "real",
            "integer",
            "natural",
            "rational",
            "irrational",
            "prime",
            "composite",
            "factor",
            "multiple",
            "divisor",
            "remainder",
            "quotient",
            "sum",
            "difference",
            "product",
            "power",
            "root",
            "logarithm",
            "exponential",
            "sine",
            "cosine",
            "tangent",
            "secant",
            "cosecant",
            "cotangent",
            "theorem",
            "proof",
            "axiom",
            "postulate",
            "lemma",
            "corollary",
            "conjecture",
            "hypothesis",
        }

        self._writing_keywords = {
            "write",
            "draft",
            "compose",
            "create",
            "generate",
            "author",
            "story",
            "narrative",
            "plot",
            "character",
            "dialogue",
            "script",
            "screenplay",
            "poem",
            "poetry",
            "haiku",
            "sonnet",
            "limerick",
            "essay",
            "article",
            "blog post",
            "content",
            "copy",
            "copywriting",
            "marketing",
            "sales",
            "pitch",
            "proposal",
            "report",
            "whitepaper",
            "case study",
            "press release",
            "announcement",
            "newsletter",
            "email",
            "letter",
            "memo",
            "note",
            "message",
            "tweet",
            "post",
            "caption",
            "headline",
            "title",
            "subtitle",
            "introduction",
            "conclusion",
            "body",
            "paragraph",
            "sentence",
            "word",
            "vocabulary",
            "grammar",
            "spelling",
            "punctuation",
            "style",
            "tone",
            "voice",
            "creative",
            "fiction",
            "non-fiction",
            "biography",
            "autobiography",
            "memoir",
            "journal",
            "diary",
        }

        self._chat_keywords = {
            "hello",
            "hi",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "how's it going",
            "what's up",
            "what's new",
            "tell me about",
            "what do you know",
            "can you help",
            "i need help",
            "assist me",
            "support",
            "guidance",
            "advice",
            "recommendation",
            "suggestion",
            "opinion",
            "thoughts",
            "perspective",
            "view",
            "stand",
            "position",
            "stance",
            "attitude",
            "feeling",
            "emotion",
            "sentiment",
            "mood",
            "atmosphere",
            "vibe",
            "energy",
            "aura",
            "personality",
            "character",
            "trait",
            "quality",
            "attribute",
            "feature",
            "aspect",
            "element",
            "component",
            "part",
            "piece",
            "segment",
            "section",
            "portion",
            "share",
            "slice",
            "chunk",
            "bit",
            "fragment",
            "shard",
            "particle",
            "speck",
            "dot",
            "point",
            "spot",
            "mark",
            "sign",
            "symbol",
            "token",
            "badge",
            "emblem",
            "icon",
            "logo",
            "brand",
            "identity",
            "image",
            "reputation",
            "name",
            "title",
            "label",
            "tag",
            "category",
            "class",
            "type",
            "kind",
            "sort",
            "variety",
            "species",
            "breed",
            "strain",
            "cultivar",
            "variety",
        }

    def classify(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None
    ) -> ClassificationResult:
        """Classify a query based on messages and tools."""
        if not messages:
            return ClassificationResult(
                query_type=QueryType.GENERAL,
                confidence=0.5,
                reasoning="No messages provided, defaulting to general",
            )

        text = self._extract_text_from_messages(messages)
        if not text:
            return ClassificationResult(
                query_type=QueryType.GENERAL,
                confidence=0.5,
                reasoning="No text content found, defaulting to general",
            )

        scores = {
            QueryType.CODE: self._score_for_query_type(text, self._code_keywords),
            QueryType.VISION: self._score_for_query_type(text, self._vision_keywords),
            QueryType.SUMMARIZATION: self._score_for_query_type(
                text, self._summarization_keywords
            ),
            QueryType.REASONING: self._score_for_query_type(
                text, self._reasoning_keywords
            ),
            QueryType.MATH: self._score_for_query_type(text, self._math_keywords),
            QueryType.WRITING: self._score_for_query_type(text, self._writing_keywords),
            QueryType.CHAT: self._score_for_query_type(text, self._chat_keywords),
        }

        if tools:
            scores[QueryType.CODE] += 0.3

        has_images = self._has_image_content(messages)
        if has_images:
            scores[QueryType.VISION] += 0.5

        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        if best_score < 0.1:
            best_type = QueryType.GENERAL
            reasoning = "Low confidence scores, defaulting to general"
        else:
            reasoning = f"Best match: {best_type.value} (score: {best_score:.2f})"

        logger.debug(
            "QUERY_CLASSIFIER: type={} confidence={} reasoning='{}'",
            best_type,
            best_score,
            reasoning,
        )

        return ClassificationResult(
            query_type=best_type,
            confidence=best_score,
            reasoning=reasoning,
        )

    def _extract_text_from_messages(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Extract all text content from messages."""
        text_parts: list[str] = []

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            text_parts.append("[IMAGE]")

        return " ".join(text_parts)

    def _has_image_content(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        """Check if any message contains image content."""
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        return True
        return False

    def _score_for_query_type(self, text: str, keywords: set[str]) -> float:
        """Calculate a score for a query type based on keyword matches."""
        text_lower = text.lower()
        word_count = len(text_lower.split())

        if word_count == 0:
            return 0.0

        matches = 0
        for keyword in keywords:
            if keyword in text_lower:
                matches += 1

        if matches == 0:
            return 0.0

        # Use matches per word as the primary score rather than matches per total possible keywords
        # This prevents large keyword sets from causing artificially low scores
        base_score = matches / word_count
        
        # Boost confidence slightly for longer queries that have matches
        length_boost = min(0.5, word_count / 20.0) if word_count > 5 else 0.0
        
        final_score = base_score + (base_score * length_boost)
        return min(1.0, final_score)


_global_classifier: QueryClassifier | None = None


def get_global_classifier() -> QueryClassifier:
    """Get or create the global query classifier instance."""
    global _global_classifier
    if _global_classifier is None:
        _global_classifier = QueryClassifier()
    return _global_classifier
