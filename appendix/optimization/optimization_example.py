"""
LLM Optimization Techniques
============================
Practical examples of optimization techniques for LLM applications.
"""

import time
import hashlib
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# TECHNIQUE 1: SEMANTIC CACHING
# ============================================================================

class SemanticCache:
    """
    Cache responses based on semantic similarity.
    Reduces costs by 30-70% for repetitive queries.
    """
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: Dict[str, Dict] = {}
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
    
    def _simple_hash(self, text: str) -> str:
        """Create a hash for exact matching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Remove extra whitespace, lowercase, remove punctuation
        text = ' '.join(text.lower().split())
        return text
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response if similar query exists."""
        normalized = self._normalize_text(query)
        
        # Check exact match first (fast)
        exact_key = self._simple_hash(normalized)
        if exact_key in self.cache:
            self.hits += 1
            return self.cache[exact_key]["response"]
        
        # Check fuzzy match (slower but catches variations)
        for key, entry in self.cache.items():
            # Simple word overlap as proxy for semantic similarity
            query_words = set(normalized.split())
            cached_words = set(entry["normalized_query"].split())
            
            if not query_words or not cached_words:
                continue
                
            overlap = len(query_words.intersection(cached_words))
            similarity = overlap / max(len(query_words), len(cached_words))
            
            if similarity >= self.threshold:
                self.hits += 1
                print(f"  [CACHE HIT] Similarity: {similarity:.2f}")
                return entry["response"]
        
        self.misses += 1
        return None
    
    def set(self, query: str, response: str):
        """Cache a query-response pair."""
        normalized = self._normalize_text(query)
        key = self._simple_hash(normalized)
        
        self.cache[key] = {
            "query": query,
            "normalized_query": normalized,
            "response": response,
            "timestamp": time.time()
        }
    
    def stats(self) -> Dict:
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_size": len(self.cache)
        }


# ============================================================================
# TECHNIQUE 2: PROMPT OPTIMIZATION
# ============================================================================

def optimize_prompt(prompt: str) -> str:
    """
    Optimize prompt for token efficiency.
    
    Strategies:
    - Remove filler words
    - Use direct language
    - Eliminate redundancy
    """
    # Common filler phrases to remove
    fillers = [
        "i would like you to",
        "please help me to",
        "can you please",
        "i want you to",
        "could you help me",
        "i need you to"
    ]
    
    optimized = prompt.lower()
    
    # Remove fillers
    for filler in fillers:
        optimized = optimized.replace(filler, "")
    
    # Clean up extra spaces
    optimized = ' '.join(optimized.split())
    
    # Remove unnecessary words
    unnecessary = ["actually", "basically", "literally", "just", "really"]
    for word in unnecessary:
        optimized = optimized.replace(f" {word} ", " ")
    
    return optimized.strip()


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters)."""
    return len(text) // 4


# ============================================================================
# TECHNIQUE 3: MODEL ROUTING
# ============================================================================

@dataclass
class ModelConfig:
    name: str
    cost_per_1m: float
    speed_multiplier: float
    capability_level: int  # 1=basic, 2=medium, 3=advanced


class ModelRouter:
    """
    Route requests to appropriate models based on task complexity.
    Can reduce costs by 50-80%.
    """
    
    def __init__(self):
        self.models = {
            "tiny": ModelConfig("TinyLLM", 0.10, 5.0, 1),
            "small": ModelConfig("SmallLLM", 0.50, 3.0, 1),
            "medium": ModelConfig("MediumLLM", 2.00, 1.5, 2),
            "large": ModelConfig("LargeLLM", 30.00, 1.0, 3)
        }
        self.routing_log = []
    
    def analyze_complexity(self, query: str) -> Dict:
        """Analyze query to determine required model capability."""
        score = 0
        reasons = []
        
        # Length indicator
        if len(query) > 200:
            score += 2
            reasons.append("long query")
        elif len(query) > 100:
            score += 1
            reasons.append("medium length")
        
        # Keywords indicating complexity
        complex_keywords = [
            "analyze", "compare", "explain why", "how does",
            "calculate", "solve", "optimize", "design"
        ]
        
        for keyword in complex_keywords:
            if keyword in query.lower():
                score += 1
                reasons.append(f"contains '{keyword}'")
        
        # Code-related
        if any(x in query.lower() for x in ["code", "function", "program", "debug"]):
            score += 1
            reasons.append("coding task")
        
        # Math/reasoning
        if any(x in query for x in ["=", "+", "-", "*", "/"]) or "calculate" in query.lower():
            score += 2
            reasons.append("mathematical")
        
        return {"score": score, "reasons": reasons}
    
    def route(self, query: str) -> str:
        """Select appropriate model for the query."""
        analysis = self.analyze_complexity(query)
        score = analysis["score"]
        
        # Routing logic
        if score == 0:
            model = "tiny"
        elif score <= 2:
            model = "small"
        elif score <= 4:
            model = "medium"
        else:
            model = "large"
        
        self.routing_log.append({
            "query": query[:50],
            "score": score,
            "selected_model": model,
            "reasons": analysis["reasons"]
        })
        
        return model
    
    def get_savings(self) -> Dict:
        """Calculate cost savings from routing."""
        if not self.routing_log:
            return {"savings": 0, "percentage": 0}
        
        # Calculate what it would cost if all used large model
        large_cost = len(self.routing_log) * self.models["large"].cost_per_1m
        
        # Calculate actual cost with routing
        actual_cost = sum(
            self.models[entry["selected_model"]].cost_per_1m
            for entry in self.routing_log
        )
        
        savings = large_cost - actual_cost
        percentage = (savings / large_cost * 100) if large_cost > 0 else 0
        
        return {
            "baseline_cost": f"${large_cost:.4f}",
            "actual_cost": f"${actual_cost:.4f}",
            "savings": f"${savings:.4f}",
            "savings_percentage": f"{percentage:.1f}%"
        }


# ============================================================================
# TECHNIQUE 4: RESPONSE LENGTH CONTROL
# ============================================================================

def suggest_max_tokens(task_type: str) -> int:
    """Suggest appropriate max_tokens based on task type."""
    limits = {
        "greeting": 30,
        "classification": 10,
        "extraction": 50,
        "summary_short": 100,
        "summary_long": 300,
        "qa_simple": 100,
        "qa_complex": 300,
        "creative_writing": 500,
        "code_generation": 500,
        "analysis": 400,
        "translation": 200
    }
    return limits.get(task_type, 200)


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_semantic_caching():
    """Demonstrate semantic caching."""
    print("\n" + "="*60)
    print("TECHNIQUE 1: Semantic Caching")
    print("="*60)
    
    cache = SemanticCache(similarity_threshold=0.7)
    
    test_queries = [
        ("What is machine learning?", "ML is a subset of AI..."),
        ("What's ML?", "ML is a subset of AI..."),  # Should hit cache
        ("Explain neural networks", "Neural networks are..."),
        ("How do neural nets work?", "Neural networks are..."),  # Should hit cache
        ("What's the weather?", "I don't have real-time data..."),
        ("Tell me about weather", "I don't have real-time data..."),  # Should hit cache
    ]
    
    for query, response in test_queries:
        print(f"\nQuery: '{query}'")
        cached = cache.get(query)
        if cached:
            print(f"  → Returned cached response")
        else:
            print(f"  → Generated new response")
            cache.set(query, response)
    
    print(f"\nCache Statistics: {cache.stats()}")


def demo_prompt_optimization():
    """Demonstrate prompt optimization."""
    print("\n" + "="*60)
    print("TECHNIQUE 2: Prompt Optimization")
    print("="*60)
    
    examples = [
        "I would like you to please help me write a function",
        "Can you please explain to me how machine learning works",
        "I actually just need you to basically summarize this text",
        "Could you help me to compare these two approaches"
    ]
    
    print(f"{'Original':<60} {'Optimized':<40} {'Savings'}")
    print("-" * 100)
    
    for prompt in examples:
        optimized = optimize_prompt(prompt)
        original_tokens = estimate_tokens(prompt)
        optimized_tokens = estimate_tokens(optimized)
        savings = ((original_tokens - optimized_tokens) / original_tokens * 100)
        
        print(f"{prompt:<60} {optimized:<40} {savings:.0f}% fewer tokens")


def demo_model_routing():
    """Demonstrate intelligent model routing."""
    print("\n" + "="*60)
    print("TECHNIQUE 3: Model Routing")
    print("="*60)
    
    router = ModelRouter()
    
    test_queries = [
        "Hi!",
        "What's 2+2?",
        "Summarize this article in one sentence",
        "Compare and contrast supervised vs unsupervised learning",
        "Write a Python function to sort a list",
        "Design a scalable architecture for a social media platform"
    ]
    
    print(f"{'Query':<50} {'Model':<10} {'Reason'}")
    print("-" * 80)
    
    for query in test_queries:
        model = router.route(query)
        analysis = router.analyze_complexity(query)
        reason = analysis["reasons"][0] if analysis["reasons"] else "simple"
        print(f"{query[:50]:<50} {model:<10} {reason}")
    
    print(f"\nCost Analysis: {router.get_savings()}")


def demo_token_budgeting():
    """Demonstrate token budgeting."""
    print("\n" + "="*60)
    print("TECHNIQUE 4: Token Budgeting")
    print("="*60)
    
    tasks = [
        ("greeting", "Hello!"),
        ("classification", "Is this positive or negative?"),
        ("qa_simple", "What is the capital of France?"),
        ("code_generation", "Write a sorting algorithm"),
        ("analysis", "Analyze market trends and provide recommendations")
    ]
    
    print(f"{'Task Type':<20} {'Suggested Max Tokens':<25} {'Est. Cost*'}")
    print("-" * 60)
    
    for task_type, example in tasks:
        tokens = suggest_max_tokens(task_type)
        cost = tokens * 0.000002  # Approximate cost at $2/1M tokens
        print(f"{task_type:<20} {tokens:<25} ${cost:.5f}")
    
    print("\n*Estimated cost using mid-tier model ($2/1M tokens)")


def main():
    """Run all optimization demonstrations."""
    print("\n" + "="*60)
    print("LLM OPTIMIZATION TECHNIQUES")
    print("="*60)
    print("\nLearn how to make your LLM applications faster and cheaper!")
    
    demo_semantic_caching()
    demo_prompt_optimization()
    demo_model_routing()
    demo_token_budgeting()
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. Semantic Caching: Save 30-70% on repetitive queries
2. Prompt Optimization: Reduce tokens by 20-50%
3. Model Routing: Save 50-80% by right-sizing models
4. Token Budgeting: Control costs with appropriate limits

Combined, these techniques can reduce LLM costs by 60-80%
while improving response times!
""")


if __name__ == "__main__":
    main()
