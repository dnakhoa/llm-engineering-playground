"""
LLM Ops & Observability - Production Monitoring Example

This module demonstrates production-ready observability patterns:
- Distributed tracing for LLM calls
- Cost tracking and budget alerts
- Quality monitoring and drift detection
- Prompt versioning and A/B testing
- Feedback loop pipelines
- Dashboard metrics collection
"""

import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class LatencyMetrics:
    total_ms: float
    ttft_ms: float  # Time to first token
    tokens_per_second: float


@dataclass
class QualityScore:
    relevance: float  # 0-1
    coherence: float  # 0-1
    factuality: float  # 0-1
    safety: float  # 0-1
    overall: float


@dataclass
class LLMSpan:
    """Represents a single LLM call with full observability"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    timestamp: datetime
    model: str
    provider: ModelProvider
    prompt: str
    prompt_version: str
    response: str
    token_usage: TokenUsage
    latency: LatencyMetrics
    quality_score: Optional[QualityScore]
    metadata: Dict[str, Any]
    user_feedback: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict:
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'timestamp': self.timestamp.isoformat(),
            'model': self.model,
            'provider': self.provider.value,
            'prompt': self.prompt,
            'prompt_version': self.prompt_version,
            'response': self.response,
            'token_usage': asdict(self.token_usage),
            'latency': asdict(self.latency),
            'quality_score': asdict(self.quality_score) if self.quality_score else None,
            'metadata': self.metadata,
            'user_feedback': self.user_feedback,
            'tags': self.tags
        }


class ObservabilityClient:
    """Simulated observability backend (replace with LangSmith, Arize, etc.)"""
    
    def __init__(self):
        self.spans: List[LLMSpan] = []
        self.traces: Dict[str, List[LLMSpan]] = {}
        self.cost_budgets: Dict[str, float] = {}
        self.alerts: List[Dict] = []
    
    def log_span(self, span: LLMSpan):
        """Log a span to the observability backend"""
        self.spans.append(span)
        
        if span.trace_id not in self.traces:
            self.traces[span.trace_id] = []
        self.traces[span.trace_id].append(span)
        
        # Check for alerts
        self._check_alerts(span)
    
    def _check_alerts(self, span: LLMSpan):
        """Check if any alert thresholds are breached"""
        # Cost alert
        if span.token_usage.cost_usd > 0.10:  # $0.10 threshold
            self.alerts.append({
                'type': 'COST_THRESHOLD',
                'severity': 'WARNING',
                'message': f"Single call cost ${span.token_usage.cost_usd:.4f} exceeds threshold",
                'span_id': span.span_id,
                'timestamp': datetime.now().isoformat()
            })
        
        # Quality alert
        if span.quality_score and span.quality_score.overall < 0.7:
            self.alerts.append({
                'type': 'LOW_QUALITY',
                'severity': 'WARNING',
                'message': f"Quality score {span.quality_score.overall:.2f} below threshold",
                'span_id': span.span_id,
                'timestamp': datetime.now().isoformat()
            })
        
        # Latency alert
        if span.latency.total_ms > 5000:  # 5 seconds
            self.alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'WARNING',
                'message': f"Latency {span.latency.total_ms:.0f}ms exceeds threshold",
                'span_id': span.span_id,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_trace(self, trace_id: str) -> List[LLMSpan]:
        """Retrieve all spans for a trace"""
        return self.traces.get(trace_id, [])
    
    def get_dashboard_metrics(self, hours: int = 24) -> Dict:
        """Generate dashboard metrics for the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_spans = [s for s in self.spans if s.timestamp > cutoff]
        
        if not recent_spans:
            return {}
        
        # Cost metrics
        total_cost = sum(s.token_usage.cost_usd for s in recent_spans)
        cost_by_model = {}
        for span in recent_spans:
            cost_by_model[span.model] = cost_by_model.get(span.model, 0) + span.token_usage.cost_usd
        
        # Latency metrics
        avg_latency = sum(s.latency.total_ms for s in recent_spans) / len(recent_spans)
        p95_latency = sorted([s.latency.total_ms for s in recent_spans])[int(len(recent_spans) * 0.95)]
        
        # Quality metrics
        quality_scores = [s.quality_score.overall for s in recent_spans if s.quality_score]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Volume metrics
        total_tokens = sum(s.token_usage.total_tokens for s in recent_spans)
        requests_per_hour = len(recent_spans) / hours
        
        # Feedback metrics
        feedback_count = sum(1 for s in recent_spans if s.user_feedback)
        positive_feedback = sum(1 for s in recent_spans if s.user_feedback and s.user_feedback.get('rating', 0) >= 4)
        feedback_rate = positive_feedback / feedback_count if feedback_count > 0 else 0
        
        return {
            'time_range_hours': hours,
            'total_requests': len(recent_spans),
            'total_cost_usd': round(total_cost, 4),
            'cost_by_model': {k: round(v, 4) for k, v in cost_by_model.items()},
            'avg_latency_ms': round(avg_latency, 2),
            'p95_latency_ms': round(p95_latency, 2),
            'avg_quality_score': round(avg_quality, 3),
            'total_tokens': total_tokens,
            'requests_per_hour': round(requests_per_hour, 2),
            'feedback_count': feedback_count,
            'positive_feedback_rate': round(feedback_rate, 3),
            'alert_count': len(self.alerts)
        }


class PromptVersionManager:
    """Manage prompt versions and A/B tests"""
    
    def __init__(self):
        self.prompts: Dict[str, Dict[str, str]] = {}  # name -> {version: template}
        self.ab_tests: Dict[str, Dict] = {}  # test_name -> config
    
    def register_prompt(self, name: str, version: str, template: str):
        """Register a new prompt version"""
        if name not in self.prompts:
            self.prompts[name] = {}
        self.prompts[name][version] = template
        print(f"✅ Registered prompt '{name}' version '{version}'")
    
    def create_ab_test(self, test_name: str, prompt_name: str, variants: List[str], 
                       traffic_split: List[float]):
        """Create an A/B test between prompt variants"""
        if len(variants) != len(traffic_split):
            raise ValueError("Variants and traffic_split must have same length")
        
        self.ab_tests[test_name] = {
            'prompt_name': prompt_name,
            'variants': variants,
            'traffic_split': traffic_split,
            'results': {v: {'count': 0, 'avg_quality': 0} for v in variants}
        }
        print(f"🧪 Created A/B test '{test_name}' with variants: {variants}")
    
    def get_variant(self, test_name: str, request_id: str) -> str:
        """Get variant for a request based on traffic split"""
        test = self.ab_tests[test_name]
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100
        
        cumulative = 0
        for variant, split in zip(test['variants'], test['traffic_split']):
            cumulative += split * 100
            if hash_val < cumulative:
                return variant
        return test['variants'][-1]
    
    def record_result(self, test_name: str, variant: str, quality_score: float):
        """Record result for an A/B test variant"""
        test = self.ab_tests[test_name]
        current = test['results'][variant]
        count = current['count']
        avg = current['avg_quality']
        
        # Update running average
        new_avg = ((avg * count) + quality_score) / (count + 1)
        test['results'][variant] = {'count': count + 1, 'avg_quality': new_avg}
    
    def get_ab_results(self, test_name: str) -> Dict:
        """Get A/B test results"""
        return self.ab_tests.get(test_name, {})


class FeedbackCollector:
    """Collect and process user feedback"""
    
    def __init__(self, obs_client: ObservabilityClient):
        self.obs_client = obs_client
        self.feedback_queue: List[Dict] = []
    
    def submit_feedback(self, span_id: str, rating: int, comment: str = "", 
                        tags: List[str] = None):
        """Submit user feedback for a span"""
        feedback = {
            'span_id': span_id,
            'rating': rating,  # 1-5
            'comment': comment,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_queue.append(feedback)
        
        # Update the span
        for span in self.obs_client.spans:
            if span.span_id == span_id:
                span.user_feedback = feedback
                break
        
        print(f"⭐ Feedback submitted for span {span_id[:8]}...: {rating}/5")
    
    def get_feedback_summary(self, hours: int = 24) -> Dict:
        """Get feedback summary"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_feedback = [
            f for f in self.feedback_queue 
            if datetime.fromisoformat(f['timestamp']) > cutoff
        ]
        
        if not recent_feedback:
            return {}
        
        ratings = [f['rating'] for f in recent_feedback]
        avg_rating = sum(ratings) / len(ratings)
        
        return {
            'count': len(recent_feedback),
            'avg_rating': round(avg_rating, 2),
            'distribution': {i: ratings.count(i) for i in range(1, 6)},
            'common_tags': self._get_common_tags(recent_feedback)
        }
    
    def _get_common_tags(self, feedback_list: List[Dict]) -> List[str]:
        all_tags = []
        for f in feedback_list:
            all_tags.extend(f.get('tags', []))
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return sorted(tag_counts.keys(), key=lambda x: tag_counts[x], reverse=True)[:5]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def simulate_llm_call(obs_client: ObservabilityClient, prompt: str, 
                      prompt_version: str, model: str = "gpt-4") -> LLMSpan:
    """Simulate an LLM call with realistic metrics"""
    
    trace_id = str(uuid.uuid4())
    span_id = str(uuid.uuid4())
    
    # Simulate processing
    time.sleep(0.1)  # Simulate network latency
    
    # Generate mock response
    response = f"This is a simulated response to: {prompt[:50]}..."
    
    # Calculate mock metrics
    prompt_tokens = len(prompt) // 4
    completion_tokens = len(response) // 4
    cost_per_token = 0.00003  # Approximate GPT-4 cost
    
    span = LLMSpan(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        timestamp=datetime.now(),
        model=model,
        provider=ModelProvider.OPENAI,
        prompt=prompt,
        prompt_version=prompt_version,
        response=response,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=(prompt_tokens + completion_tokens) * cost_per_token
        ),
        latency=LatencyMetrics(
            total_ms=1250.5,
            ttft_ms=320.2,
            tokens_per_second=45.3
        ),
        quality_score=QualityScore(
            relevance=0.92,
            coherence=0.88,
            factuality=0.85,
            safety=0.98,
            overall=0.91
        ),
        metadata={
            'user_id': 'user_123',
            'session_id': 'session_456',
            'application': 'customer_support'
        },
        tags=['production', 'customer-facing']
    )
    
    obs_client.log_span(span)
    return span


def main():
    print("=" * 70)
    print("LLM OPS & OBSERVABILITY - PRODUCTION MONITORING DEMO")
    print("=" * 70)
    
    # Initialize components
    obs_client = ObservabilityClient()
    prompt_manager = PromptVersionManager()
    feedback_collector = FeedbackCollector(obs_client)
    
    # Register prompt versions
    print("\n📝 Registering prompt versions...")
    prompt_manager.register_prompt(
        "customer_support_greeting",
        "v1",
        "You are a helpful support agent. Greet the customer and ask how you can help."
    )
    prompt_manager.register_prompt(
        "customer_support_greeting",
        "v2",
        "You are a friendly support specialist. Start with a warm greeting and offer assistance."
    )
    
    # Create A/B test
    print("\n🧪 Setting up A/B test...")
    prompt_manager.create_ab_test(
        "greeting_test",
        "customer_support_greeting",
        ["v1", "v2"],
        [0.5, 0.5]
    )
    
    # Simulate production traffic
    print("\n🚀 Simulating production traffic...")
    prompts = [
        "How do I reset my password?",
        "What's your refund policy?",
        "Can I upgrade my plan?",
        "I'm having trouble logging in",
        "How do I contact support?"
    ]
    
    for i, prompt in enumerate(prompts):
        # Get A/B test variant
        variant = prompt_manager.get_variant("greeting_test", f"req_{i}")
        
        # Simulate call
        span = simulate_llm_call(
            obs_client,
            prompt,
            prompt_version=variant,
            model="gpt-4"
        )
        
        # Record A/B test result
        prompt_manager.record_result("greeting_test", variant, span.quality_score.overall)
        
        # Simulate some user feedback
        if i % 2 == 0:
            rating = 5 if i % 4 == 0 else 4
            feedback_collector.submit_feedback(
                span.span_id,
                rating=rating,
                comment="Very helpful!" if rating == 5 else "Good response",
                tags=['helpful', 'quick'] if rating == 5 else ['okay']
            )
    
    # Simulate some problematic calls
    print("\n⚠️ Simulating edge cases...")
    
    # High cost call
    span = simulate_llm_call(
        obs_client,
        "Analyze this 50000 word document...",
        "v1",
        model="gpt-4-turbo"
    )
    span.token_usage.cost_usd = 0.15  # Force over threshold
    span.token_usage.total_tokens = 5000
    obs_client.log_span(span)
    
    # Low quality call
    span = simulate_llm_call(
        obs_client,
        "Random gibberish query xyz123",
        "v1"
    )
    span.quality_score.overall = 0.65
    span.quality_score.relevance = 0.60
    obs_client.log_span(span)
    
    # Show dashboard metrics
    print("\n📊 DASHBOARD METRICS (Last 24 hours)")
    print("-" * 70)
    metrics = obs_client.get_dashboard_metrics()
    
    print(f"Total Requests:      {metrics['total_requests']}")
    print(f"Total Cost:          ${metrics['total_cost_usd']:.4f}")
    print(f"Avg Latency:         {metrics['avg_latency_ms']:.2f}ms")
    print(f"P95 Latency:         {metrics['p95_latency_ms']:.2f}ms")
    print(f"Avg Quality Score:   {metrics['avg_quality_score']:.3f}")
    print(f"Total Tokens:        {metrics['total_tokens']:,}")
    print(f"Requests/Hour:       {metrics['requests_per_hour']:.2f}")
    print(f"Feedback Count:      {metrics['feedback_count']}")
    print(f"Positive Feedback:   {metrics['positive_feedback_rate']:.1%}")
    print(f"Active Alerts:       {metrics['alert_count']}")
    
    # Show cost breakdown
    print("\n💰 COST BY MODEL")
    print("-" * 70)
    for model, cost in metrics['cost_by_model'].items():
        print(f"{model:20} ${cost:.4f}")
    
    # Show alerts
    print("\n🚨 ACTIVE ALERTS")
    print("-" * 70)
    for alert in obs_client.alerts[:5]:
        print(f"[{alert['severity']}] {alert['type']}: {alert['message']}")
    
    # Show A/B test results
    print("\n🧪 A/B TEST RESULTS")
    print("-" * 70)
    ab_results = prompt_manager.get_ab_results("greeting_test")
    print(f"Test: greeting_test")
    print(f"Variants: {ab_results['variants']}")
    print(f"Traffic Split: {ab_results['traffic_split']}")
    print("\nResults:")
    for variant, data in ab_results['results'].items():
        print(f"  {variant}: {data['count']} requests, avg quality: {data['avg_quality']:.3f}")
    
    # Show feedback summary
    print("\n⭐ FEEDBACK SUMMARY")
    print("-" * 70)
    feedback_summary = feedback_collector.get_feedback_summary()
    print(f"Total Feedback:      {feedback_summary['count']}")
    print(f"Average Rating:      {feedback_summary['avg_rating']:.2f}/5")
    print(f"Rating Distribution: {feedback_summary['distribution']}")
    print(f"Common Tags:         {feedback_summary['common_tags']}")
    
    # Show trace example
    print("\n🔍 TRACE EXAMPLE")
    print("-" * 70)
    if obs_client.traces:
        trace_id = list(obs_client.traces.keys())[0]
        trace = obs_client.get_trace(trace_id)
        print(f"Trace ID: {trace_id[:8]}...")
        print(f"Spans in trace: {len(trace)}")
        for span in trace:
            print(f"  - {span.model}: {span.prompt[:40]}... (quality: {span.quality_score.overall:.2f})")
    
    print("\n" + "=" * 70)
    print("✅ OBSERVABILITY DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Track every LLM call with detailed spans")
    print("2. Monitor cost, latency, and quality in real-time")
    print("3. Set up alerts for threshold breaches")
    print("4. Use A/B testing for prompt optimization")
    print("5. Collect user feedback for continuous improvement")
    print("6. Build dashboards for team visibility")


if __name__ == "__main__":
    main()
