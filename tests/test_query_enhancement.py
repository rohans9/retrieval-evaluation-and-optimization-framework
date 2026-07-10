"""Query enhancement behavior tests."""

from __future__ import annotations

from retrieval_evaluation_framework.config.settings import QueryEnhancementConfig
from retrieval_evaluation_framework.query_enhancement.factory import QueryEnhancerFactory


def test_query_enhancement_factory_builds_sequential_chain() -> None:
    config = QueryEnhancementConfig.model_validate(
        {
            "enabled": True,
            "methods": ["rewrite", "hyde"],
            "rewrite": {"backend": "template"},
            "hyde": {"backend": "template"},
        }
    )

    enhancer = QueryEnhancerFactory.create(config, embedding_engine=None)

    assert enhancer is not None
    result = enhancer.enhance("What is maternity leave policy?")
    assert result.method == "rewrite+hyde"
    assert result.enhanced_query
    assert result.metadata.get("steps") is not None
