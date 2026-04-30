"""Tests for auto-routing functionality."""

import pytest
from core.model_catalog import ModelCatalog, ModelInfo
from core.query_classifier import QueryClassifier, ClassificationResult, QueryType


class TestModelCatalog:
    """Test model catalog functionality."""

    def test_model_info_creation(self):
        """Test creating a ModelInfo instance."""
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider_id="nvidia_nim",
            capabilities=frozenset([QueryType.CODE, QueryType.CHAT]),
            context_length=128000,
            is_free=True,
            description="A test model",
        )
        assert model.id == "test-model"
        assert model.provider_id == "nvidia_nim"
        assert QueryType.CODE in model.capabilities
        assert QueryType.CHAT in model.capabilities
        assert model.context_length == 128000
        assert model.is_free is True

    def test_catalog_cache_validity(self):
        """Test cache validity checking."""
        catalog = ModelCatalog(cache_ttl=300.0)
        assert not catalog.is_cache_valid()

    def test_get_model_by_id(self):
        """Test retrieving a model by ID."""
        catalog = ModelCatalog()
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider_id="nvidia_nim",
            capabilities=frozenset([QueryType.CODE]),
        )
        catalog._update_cache([model])

        retrieved = catalog.get_model("test-model")
        assert retrieved is not None
        assert retrieved.id == "test-model"

    def test_get_models_for_capability(self):
        """Test retrieving models by capability."""
        catalog = ModelCatalog()
        models = [
            ModelInfo(
                id="code-model",
                name="Code Model",
                provider_id="nvidia_nim",
                capabilities=frozenset([QueryType.CODE]),
            ),
            ModelInfo(
                id="chat-model",
                name="Chat Model",
                provider_id="nvidia_nim",
                capabilities=frozenset([QueryType.CHAT]),
            ),
            ModelInfo(
                id="general-model",
                name="General Model",
                provider_id="nvidia_nim",
                capabilities=frozenset([QueryType.CODE, QueryType.CHAT]),
            ),
        ]
        catalog._update_cache(models)

        code_models = catalog.get_models_for_capability(QueryType.CODE)
        assert len(code_models) == 2

        chat_models = catalog.get_models_for_capability(QueryType.CHAT)
        assert len(chat_models) == 2

    def test_find_best_model_for_query(self):
        """Test finding the best model for a query type."""
        catalog = ModelCatalog()
        models = [
            ModelInfo(
                id="small-code-model",
                name="Small Code Model",
                provider_id="nvidia_nim",
                capabilities=frozenset([QueryType.CODE]),
                context_length=32000,
            ),
            ModelInfo(
                id="large-code-model",
                name="Large Code Model",
                provider_id="nvidia_nim",
                capabilities=frozenset([QueryType.CODE]),
                context_length=128000,
            ),
        ]
        catalog._update_cache(models)

        best = catalog.find_best_model_for_query(QueryType.CODE)
        assert best is not None
        assert best.id == "large-code-model"

    def test_clear_cache(self):
        """Test clearing the model cache."""
        catalog = ModelCatalog()
        model = ModelInfo(
            id="test-model",
            name="Test Model",
            provider_id="nvidia_nim",
            capabilities=frozenset([QueryType.CODE]),
        )
        catalog._update_cache([model])

        assert catalog.get_model("test-model") is not None
        catalog.clear_cache()
        assert catalog.get_model("test-model") is None


class TestQueryClassifier:
    """Test query classifier functionality."""

    def test_classify_code_query(self):
        """Test classifying a code-related query."""
        classifier = QueryClassifier()
        messages = [{"role": "user", "content": "Write a Python function to sort a list"}]

        result = classifier.classify(messages)
        assert result.query_type == QueryType.CODE
        assert result.confidence > 0

    def test_classify_vision_query(self):
        """Test classifying a vision-related query."""
        classifier = QueryClassifier()
        messages = [
            {"role": "user", "content": "What do you see in this image?"}
        ]

        result = classifier.classify(messages)
        assert result.query_type == QueryType.VISION
        assert result.confidence > 0

    def test_classify_summarization_query(self):
        """Test classifying a summarization query."""
        classifier = QueryClassifier()
        messages = [
            {"role": "user", "content": "Summarize this article in 3 bullet points"}
        ]

        result = classifier.classify(messages)
        assert result.query_type == QueryType.SUMMARIZATION
        assert result.confidence > 0

    def test_classify_with_tools(self):
        """Test classifying a query with tools present."""
        classifier = QueryClassifier()
        messages = [{"role": "user", "content": "Help me with this task"}]
        tools = [{"name": "code_executor", "description": "Execute code"}]

        result = classifier.classify(messages, tools)
        assert result.query_type == QueryType.CODE
        assert result.confidence > 0

    def test_classify_with_image_content(self):
        """Test classifying a query with image content."""
        classifier = QueryClassifier()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                ],
            }
        ]

        result = classifier.classify(messages)
        assert result.query_type == QueryType.VISION
        assert result.confidence > 0

    def test_classify_empty_messages(self):
        """Test classifying with no messages."""
        classifier = QueryClassifier()
        result = classifier.classify([])

        assert result.query_type == QueryType.GENERAL
        assert result.confidence == 0.5

    def test_classify_low_confidence_fallback(self):
        """Test that low confidence queries fall back to GENERAL."""
        classifier = QueryClassifier()
        messages = [{"role": "user", "content": "xyz"}]

        result = classifier.classify(messages)
        assert result.query_type == QueryType.GENERAL


class TestQueryType:
    """Test QueryType enum."""

    def test_query_type_values(self):
        """Test that QueryType has expected values."""
        assert QueryType.CODE.value == "code"
        assert QueryType.CHAT.value == "chat"
        assert QueryType.SUMMARIZATION.value == "summarization"
        assert QueryType.VISION.value == "vision"
        assert QueryType.GENERAL.value == "general"
        assert QueryType.REASONING.value == "reasoning"
        assert QueryType.MATH.value == "math"
        assert QueryType.WRITING.value == "writing"
