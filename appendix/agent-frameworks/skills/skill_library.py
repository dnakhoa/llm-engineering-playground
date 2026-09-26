"""
Skills Library - Reusable capabilities for agents

Skills are atomic, composable functions that can be used as tools by agents.
Each skill should be:
- Well-documented with clear input/output
- Error-handled and safe
- Observable with logging
- Testable in isolation
"""

from langchain_core.tools import tool  # langchain.tools is deprecated
from typing import List, Dict, Optional, Any
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SEARCH SKILLS
# ============================================================================

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for current information.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        Formatted search results as a string
    
    Example:
        >>> web_search("latest advances in quantum computing", num_results=3)
    """
    logger.info(f"Executing web_search: {query}")
    # Mock implementation - replace with actual API (e.g., Tavily, SerpAPI)
    results = [
        f"Result 1 for: {query}",
        f"Result 2 for: {query}",
        f"Result 3 for: {query}"
    ][:num_results]
    return "\n".join(results)


@tool
def search_academic_papers(query: str, year_range: Optional[tuple] = None) -> str:
    """
    Search academic papers and research publications.
    
    Args:
        query: Research topic or keywords
        year_range: Tuple of (start_year, end_year) for filtering
    
    Returns:
        List of relevant papers with citations
    
    Example:
        >>> search_academic_papers("transformer architectures", year_range=(2020, 2024))
    """
    logger.info(f"Searching academic papers: {query}, years: {year_range}")
    # Mock implementation - replace with Semantic Scholar, arXiv API
    papers = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "year": 2017,
            "citations": 50000
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Devlin et al."],
            "year": 2018,
            "citations": 40000
        }
    ]
    
    if year_range:
        papers = [p for p in papers if year_range[0] <= p["year"] <= year_range[1]]
    
    formatted = []
    for paper in papers:
        formatted.append(
            f"{paper['title']} ({paper['year']}) - {paper['authors'][0]} "
            f"[Cited: {paper['citations']}]"
        )
    
    return "\n".join(formatted) if formatted else "No papers found."


@tool
def search_code_repository(query: str, language: str = "python") -> str:
    """
    Search code repositories for examples and implementations.
    
    Args:
        query: Code search query (functionality, algorithm, etc.)
        language: Programming language filter
    
    Returns:
        Code snippets and repository links
    
    Example:
        >>> search_code_repository("binary search implementation", language="python")
    """
    logger.info(f"Searching code: {query} in {language}")
    # Mock implementation - replace with GitHub API, StackOverflow
    return f"""
Found code examples for '{query}' in {language}:

1. Repository: github.com/example/binary-search
   - Clean implementation with tests
   - Language: {language}
   - Stars: 1.2k

2. Repository: github.com/example/algorithms
   - Part of larger algorithms library
   - Language: {language}
   - Stars: 5.4k

Example snippet:
```{language}
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
"""


# ============================================================================
# CODE EXECUTION SKILLS
# ============================================================================

@tool
def execute_python_code(code: str, timeout: int = 30) -> str:
    """
    Safely execute Python code in a sandboxed environment.
    
    ⚠️ SECURITY: In production, use proper sandboxing (Docker, restricted eval)
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
    
    Returns:
        Execution output or error message
    
    Example:
        >>> execute_python_code("print(2 + 2)")
        '4'
    """
    logger.info(f"Executing Python code (length: {len(code)})")
    
    # Basic safety checks (NOT sufficient for production!)
    dangerous_patterns = ["import os", "import sys", "__import__", "exec(", "eval("]
    for pattern in dangerous_patterns:
        if pattern in code:
            return f"❌ Security violation: Dangerous pattern detected '{pattern}'"
    
    try:
        # Mock execution - in production use subprocess with restrictions
        # For demo purposes, we'll just simulate
        if "print(" in code:
            # Extract print statement content
            start = code.find('print("') + 7
            end = code.find('")', start)
            if start > 6 and end > start:
                return code[start:end]
        return "Code executed successfully (mock output)"
    except Exception as e:
        return f"❌ Execution error: {str(e)}"


@tool
def validate_code_syntax(code: str, language: str = "python") -> Dict[str, Any]:
    """
    Validate code syntax without execution.
    
    Args:
        code: Code to validate
        language: Programming language
    
    Returns:
        Dictionary with validation results
    
    Example:
        >>> validate_code_syntax("def foo(): return 1", language="python")
        {'valid': True, 'errors': []}
    """
    logger.info(f"Validating {language} syntax")
    
    if language == "python":
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True, "errors": [], "warnings": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"Line {e.lineno}: {e.msg}"],
                "warnings": []
            }
    
    return {"valid": True, "errors": [], "warnings": [f"{language} validation not fully implemented"]}


@tool
def generate_unit_tests(function_name: str, function_code: str) -> str:
    """
    Generate unit tests for a given function.
    
    Args:
        function_name: Name of the function to test
        function_code: The function's source code
    
    Returns:
        Complete unit test code using pytest
    
    Example:
        >>> generate_unit_tests("add", "def add(a, b): return a + b")
    """
    logger.info(f"Generating tests for {function_name}")
    
    return f'''
import pytest
from your_module import {function_name}

class Test{function_name.capitalize()}:
    def test_basic_case(self):
        # TODO: Add test cases based on function behavior
        assert {function_name}(1, 2) is not None
    
    def test_edge_cases(self):
        # Test edge cases (empty inputs, None, boundaries)
        pass
    
    def test_error_handling(self):
        # Test that appropriate errors are raised
        pass
    
    # Add more specific tests based on function documentation
'''


# ============================================================================
# ANALYSIS SKILLS
# ============================================================================

@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text (positive, negative, neutral).
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary with sentiment scores and classification
    
    Example:
        >>> analyze_sentiment("I love this product!")
        {'sentiment': 'positive', 'confidence': 0.95}
    """
    logger.info(f"Analyzing sentiment (length: {len(text)})")
    
    # Mock implementation - replace with actual model
    positive_words = ["love", "great", "excellent", "amazing", "good"]
    negative_words = ["hate", "terrible", "awful", "bad", "worst"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return {"sentiment": "positive", "confidence": 0.85, "scores": {"pos": 0.85, "neg": 0.15}}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "confidence": 0.85, "scores": {"pos": 0.15, "neg": 0.85}}
    else:
        return {"sentiment": "neutral", "confidence": 0.70, "scores": {"pos": 0.50, "neg": 0.50}}


@tool
def extract_keyphrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases and topics from text.
    
    Args:
        text: Text to analyze
        max_phrases: Maximum number of phrases to return
    
    Returns:
        List of key phrases
    
    Example:
        >>> extract_keyphrases("Machine learning and AI are transforming industries...")
        ['machine learning', 'AI', 'transforming industries']
    """
    logger.info(f"Extracting keyphrases (max: {max_phrases})")
    
    # Mock implementation - replace with YAKE, RAKE, or ML model
    # Simple word frequency approach for demo
    words = text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3 and word not in ['the', 'and', 'are', 'for', 'with']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [phrase for phrase, count in sorted_phrases[:max_phrases]]


@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Generate a concise summary of text.
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length in words
    
    Returns:
        Summarized text
    
    Example:
        >>> summarize_text(long_article, max_length=50)
    """
    logger.info(f"Summarizing text (original: {len(text)} chars, max: {max_length} words)")
    
    # Mock implementation - replace with summarization model
    sentences = text.split('.')
    summary = '. '.join(sentences[:3])  # Take first 3 sentences
    return summary[:500] + "..." if len(summary) > 500 else summary


# ============================================================================
# KNOWLEDGE RETRIEVAL SKILLS
# ============================================================================

@tool
def query_knowledge_base(query: str, collection: str = "default") -> str:
    """
    Query a vector-based knowledge base for relevant information.
    
    Args:
        query: Natural language query
        collection: Knowledge base collection name
    
    Returns:
        Relevant documents/chunks from the knowledge base
    
    Example:
        >>> query_knowledge_base("What is the refund policy?", collection="customer_support")
    """
    logger.info(f"Querying knowledge base: {query} in {collection}")
    
    # Mock implementation - replace with Chroma, Pinecone, Weaviate
    return f"""
Results from '{collection}' knowledge base for query "{query}":

1. [Relevance: 0.92]
   Document: Company Policy Handbook, Section 4.2
   Content: Refunds are processed within 30 days of purchase with original receipt...

2. [Relevance: 0.87]
   Document: FAQ Database, ID: CS-104
   Content: To request a refund, contact support with your order number...

3. [Relevance: 0.81]
   Document: Terms of Service, Section 8
   Content: Digital products are non-refundable unless defective...
"""


@tool
def store_in_knowledge_base(content: str, metadata: Dict[str, Any], collection: str = "default") -> str:
    """
    Store new information in the knowledge base.
    
    Args:
        content: Text content to store
        metadata: Metadata (source, tags, category, etc.)
        collection: Target collection name
    
    Returns:
        Confirmation with document ID
    
    Example:
        >>> store_in_knowledge_base("New policy text...", {"source": "HR", "category": "policy"})
    """
    logger.info(f"Storing content in {collection} (length: {len(content)})")
    
    # Mock implementation - replace with vector DB upsert
    doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return f"✓ Stored successfully. Document ID: {doc_id}"


# ============================================================================
# UTILITY SKILLS
# ============================================================================

@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression (e.g., "2 + 2 * 3")
    
    Returns:
        Calculation result
    
    Example:
        >>> calculate("10 * (5 + 3)")
        '80'
    """
    logger.info(f"Calculating: {expression}")
    
    # Only allow safe characters
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "❌ Invalid characters in expression"
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"


@tool
def format_data(data: Any, format_type: str = "json") -> str:
    """
    Format data into specified format (JSON, CSV, Markdown table).
    
    Args:
        data: Data to format (dict, list, etc.)
        format_type: Output format (json, csv, markdown)
    
    Returns:
        Formatted data string
    
    Example:
        >>> format_data({"name": "Alice", "age": 30}, format_type="json")
    """
    logger.info(f"Formatting data as {format_type}")
    
    try:
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "csv":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                headers = list(data[0].keys())
                csv_lines = [",".join(headers)]
                for row in data:
                    csv_lines.append(",".join(str(row.get(h, "")) for h in headers))
                return "\n".join(csv_lines)
            return "❌ CSV format requires list of dicts"
        elif format_type == "markdown":
            if isinstance(data, list) and data and isinstance(data[0], dict):
                headers = list(data[0].keys())
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in data:
                    md += "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n"
                return md
            return f"```\n{json.dumps(data, indent=2, default=str)}\n```"
        else:
            return f"❌ Unsupported format: {format_type}"
    except Exception as e:
        return f"❌ Formatting error: {str(e)}"


# ============================================================================
# TOOL REGISTRY
# ============================================================================

def get_all_skills() -> list:
    """
    Get all available skills as LangChain tools.
    
    Returns:
        List of tool objects for agent initialization
    """
    return [
        web_search,
        search_academic_papers,
        search_code_repository,
        execute_python_code,
        validate_code_syntax,
        generate_unit_tests,
        analyze_sentiment,
        extract_keyphrases,
        summarize_text,
        query_knowledge_base,
        store_in_knowledge_base,
        calculate,
        format_data
    ]


def get_skills_by_category(category: str) -> list:
    """
    Get skills filtered by category.
    
    Categories: search, code, analysis, knowledge, utility
    """
    category_map = {
        "search": [web_search, search_academic_papers, search_code_repository],
        "code": [execute_python_code, validate_code_syntax, generate_unit_tests],
        "analysis": [analyze_sentiment, extract_keyphrases, summarize_text],
        "knowledge": [query_knowledge_base, store_in_knowledge_base],
        "utility": [calculate, format_data]
    }
    return category_map.get(category, [])


if __name__ == "__main__":
    # Demo usage
    print("=== Skills Library Demo ===\n")
    
    print("1. Web Search:")
    print(web_search.invoke("AI advancements 2024"))
    print("\n" + "="*50 + "\n")
    
    print("2. Code Validation:")
    result = validate_code_syntax.invoke("def hello(): return 'world'")
    print(result)
    print("\n" + "="*50 + "\n")
    
    print("3. Sentiment Analysis:")
    print(analyze_sentiment.invoke("This is absolutely amazing!"))
    print("\n" + "="*50 + "\n")
    
    print("4. Math Calculation:")
    print(calculate.invoke("15 * (8 + 2) / 3"))
    print("\n" + "="*50 + "\n")
    
    print(f"Total skills available: {len(get_all_skills())}")
