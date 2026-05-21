# Module 08: LLM Ops, Observability & Production Monitoring

## 🎯 Learning Objectives
- Implement comprehensive tracing for LLM applications
- Monitor cost, latency, and quality in production
- Set up alerting for model drift and anomalies
- Manage prompt versioning and A/B testing in production
- Build feedback loops for continuous improvement

## 📚 Core Concepts

### 1. The Three Pillars of LLM Observability
1. **Tracing**: Track every LLM call, token usage, and latency
2. **Evaluation**: Continuous quality monitoring (not just pre-deployment)
3. **Feedback**: User signals, thumbs up/down, corrections

### 2. Key Metrics to Track
| Category | Metrics |
|----------|---------|
| **Performance** | Latency (p50, p95, p99), Throughput, Error rates |
| **Cost** | Tokens in/out, Cost per request, Cost per user |
| **Quality** | Hallucination rate, Relevance score, Toxicity, Helpfulness |
| **System** | Cache hit rate, Model fallback rate, Rate limit hits |

### 3. Modern LLM Ops Stack
- **Tracing**: LangSmith, Arize Phoenix, MLflow, Weights & Biases
- **Monitoring**: Prometheus + Grafana, Datadog LLM Dashboard
- **Evaluation**: Ragas, TruLens, Custom eval pipelines
- **Feature Store**: Feast, Tecton (for retrieval features)

---

## 🛠️ Implementation Guide

### Step 1: Structured Logging & Tracing

```python
# observability/tracing.py
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class SpanType(Enum):
    LLM_CALL = "llm_call"
    RAG_RETRIEVAL = "rag_retrieval"
    TOOL_EXECUTION = "tool_execution"
    AGENT_STEP = "agent_step"
    USER_FEEDBACK = "user_feedback"

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    span_type: SpanType
    operation: str
    start_time: float
    end_time: Optional[float]
    duration_ms: Optional[float]
    status: str  # success, error, timeout
    metadata: Dict[str, Any]
    input_data: Optional[str]
    output_data: Optional[str]
    tokens: Dict[str, int]  # prompt, completion, total
    cost_usd: Optional[float]
    error_message: Optional[str]

class Tracer:
    def __init__(self, project_name: str, api_key: Optional[str] = None):
        self.project_name = project_name
        self.api_key = api_key
        self.active_spans: Dict[str, Span] = {}
        self.export_buffer: List[Span] = []
        
    def start_span(
        self,
        operation: str,
        span_type: SpanType,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        input_data: Optional[str] = None
    ) -> str:
        span_id = str(uuid.uuid4())
        trace_id = parent_span_id.split("-")[0] if parent_span_id else str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            operation=operation,
            start_time=time.time(),
            end_time=None,
            duration_ms=None,
            status="running",
            metadata=metadata or {},
            input_data=input_data,
            output_data=None,
            tokens={},
            cost_usd=None,
            error_message=None
        )
        
        self.active_spans[span_id] = span
        return span_id
    
    def end_span(
        self,
        span_id: str,
        output_data: Optional[str] = None,
        tokens: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ):
        if span_id not in self.active_spans:
            raise ValueError(f"Span {span_id} not found")
        
        span = self.active_spans[span_id]
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status
        span.output_data = output_data
        span.tokens = tokens or {}
        span.cost_usd = cost_usd
        span.error_message = error_message
        
        # Move to export buffer
        del self.active_spans[span_id]
        self.export_buffer.append(span)
        
        # Auto-export if buffer is large
        if len(self.export_buffer) >= 10:
            self.export()
    
    def export(self):
        """Export spans to backend (LangSmith, Phoenix, etc.)"""
        if not self.export_buffer:
            return
        
        # Example: Export to JSON file for local dev
        export_file = f"traces_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(export_file, "a") as f:
            for span in self.export_buffer:
                f.write(json.dumps(asdict(span)) + "\n")
        
        print(f"✅ Exported {len(self.export_buffer)} spans to {export_file}")
        self.export_buffer.clear()
    
    def record_feedback(
        self,
        trace_id: str,
        score: float,  # 0-1 or 1-5
        feedback_type: str = "user_rating",
        comment: Optional[str] = None
    ):
        """Record user feedback for a trace"""
        span_id = self.start_span(
            operation="feedback_record",
            span_type=SpanType.USER_FEEDBACK,
            metadata={
                "trace_id": trace_id,
                "score": score,
                "feedback_type": feedback_type,
                "comment": comment
            }
        )
        self.end_span(span_id, status="success")

# Global tracer instance
tracer = Tracer(project_name="llm-playground")
```

### Step 2: Cost Tracking & Budget Alerts

```python
# observability/cost_tracker.py
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CostRecord:
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    user_id: Optional[str]
    endpoint: str

# Pricing per 1K tokens (update regularly!)
MODEL_PRICING = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "llama-3-70b": {"prompt": 0.0008, "completion": 0.0008},
}

class CostTracker:
    def __init__(self):
        self.records: list[CostRecord] = []
        self.budget_alerts: Dict[str, float] = {}  # user_id -> budget
        
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        if model not in MODEL_PRICING:
            # Fallback to average pricing
            return (prompt_tokens + completion_tokens) * 0.000002
        
        pricing = MODEL_PRICING[model]
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * pricing["completion"]
        return prompt_cost + completion_cost
    
    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        user_id: Optional[str] = None,
        endpoint: str = "default"
    ) -> float:
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        record = CostRecord(
            timestamp=datetime.now(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            user_id=user_id,
            endpoint=endpoint
        )
        self.records.append(record)
        
        # Check budget alerts
        if user_id and user_id in self.budget_alerts:
            daily_spend = self.get_daily_spend(user_id)
            if daily_spend > self.budget_alerts[user_id] * 0.8:
                print(f"⚠️  WARNING: User {user_id} at 80% of budget!")
            if daily_spend > self.budget_alerts[user_id]:
                print(f"🚨 ALERT: User {user_id} exceeded budget!")
        
        return cost
    
    def get_daily_spend(self, user_id: Optional[str] = None) -> float:
        today = datetime.now().date()
        total = 0.0
        for record in self.records:
            if record.timestamp.date() == today:
                if user_id is None or record.user_id == user_id:
                    total += record.cost_usd
        return total
    
    def get_cost_breakdown(self, days: int = 7) -> Dict:
        cutoff = datetime.now() - timedelta(days=days)
        breakdown = {}
        
        for record in self.records:
            if record.timestamp < cutoff:
                continue
            
            key = record.model
            if key not in breakdown:
                breakdown[key] = {"cost": 0, "tokens": 0, "requests": 0}
            
            breakdown[key]["cost"] += record.cost_usd
            breakdown[key]["tokens"] += record.total_tokens
            breakdown[key]["requests"] += 1
        
        return breakdown

# Usage example
cost_tracker = CostTracker()

def tracked_llm_call(model, messages, user_id=None):
    # Simulate API call
    prompt_tokens = sum(len(msg["content"].split()) for msg in messages)
    completion_tokens = 150  # Estimate
    
    cost = cost_tracker.record_usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        user_id=user_id
    )
    
    print(f"💰 Cost: ${cost:.4f} | Tokens: {prompt_tokens + completion_tokens}")
    return cost
```

### Step 3: Quality Monitoring & Drift Detection

```python
# observability/quality_monitor.py
from typing import List, Dict
from dataclasses import dataclass
import statistics

@dataclass
class QualityMetric:
    name: str
    value: float
    threshold: float
    status: str  # healthy, warning, critical
    trend: str  # improving, stable, degrading

class QualityMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[str, List[float]] = {
            "hallucination_rate": [],
            "relevance_score": [],
            "toxicity_score": [],
            "helpfulness_rating": [],
            "latency_p95": []
        }
        
        self.thresholds = {
            "hallucination_rate": {"warning": 0.05, "critical": 0.10},
            "relevance_score": {"warning": 0.70, "critical": 0.60},  # Lower is worse
            "toxicity_score": {"warning": 0.30, "critical": 0.50},
            "helpfulness_rating": {"warning": 3.5, "critical": 3.0},  # Out of 5
            "latency_p95": {"warning": 2000, "critical": 5000}  # ms
        }
    
    def record_metric(self, metric_name: str, value: float):
        if metric_name not in self.metrics_history:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        history = self.metrics_history[metric_name]
        history.append(value)
        
        # Keep window size manageable
        if len(history) > self.window_size:
            history.pop(0)
    
    def check_quality(self) -> List[QualityMetric]:
        results = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Need minimum samples
                continue
            
            current_value = statistics.mean(history[-10:])  # Recent average
            
            # Determine status
            thresholds = self.thresholds[metric_name]
            if metric_name in ["relevance_score", "helpfulness_rating"]:
                # Higher is better
                if current_value < thresholds["critical"]:
                    status = "critical"
                elif current_value < thresholds["warning"]:
                    status = "warning"
                else:
                    status = "healthy"
            else:
                # Lower is better
                if current_value > thresholds["critical"]:
                    status = "critical"
                elif current_value > thresholds["warning"]:
                    status = "warning"
                else:
                    status = "healthy"
            
            # Calculate trend
            if len(history) >= 20:
                recent_avg = statistics.mean(history[-10:])
                older_avg = statistics.mean(history[-20:-10])
                
                if metric_name in ["relevance_score", "helpfulness_rating"]:
                    trend = "improving" if recent_avg > older_avg * 1.05 else "degrading" if recent_avg < older_avg * 0.95 else "stable"
                else:
                    trend = "improving" if recent_avg < older_avg * 0.95 else "degrading" if recent_avg > older_avg * 1.05 else "stable"
            else:
                trend = "stable"
            
            results.append(QualityMetric(
                name=metric_name,
                value=current_value,
                threshold=thresholds["warning"],
                status=status,
                trend=trend
            ))
        
        return results
    
    def detect_drift(self) -> Dict[str, bool]:
        """Detect if metrics have drifted significantly"""
        drift_detected = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 50:
                drift_detected[metric_name] = False
                continue
            
            # Use statistical process control (simple version)
            mean = statistics.mean(history)
            stdev = statistics.stdev(history)
            
            # Check if recent values are outside 2 sigma
            recent_values = history[-10:]
            drift_detected[metric_name] = any(
                abs(v - mean) > 2 * stdev for v in recent_values
            )
        
        return drift_detected

# Usage
monitor = QualityMonitor()

# Record metrics from evaluations
monitor.record_metric("hallucination_rate", 0.03)
monitor.record_metric("relevance_score", 0.85)
monitor.record_metric("helpfulness_rating", 4.2)
monitor.record_metric("latency_p95", 1200)

# Check quality status
quality_status = monitor.check_quality()
for metric in quality_status:
    emoji = "✅" if metric.status == "healthy" else "⚠️" if metric.status == "warning" else "🚨"
    print(f"{emoji} {metric.name}: {metric.value:.2f} ({metric.status}, {metric.trend})")
```

### Step 4: Prompt Versioning & Management

```python
# observability/prompt_registry.py
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

@dataclass
class PromptVersion:
    id: str
    name: str
    template: str
    version: str
    created_at: datetime
    created_by: str
    metadata: Dict
    performance_metrics: Dict  # avg_latency, avg_cost, avg_quality

class PromptRegistry:
    def __init__(self):
        self.prompts: Dict[str, List[PromptVersion]] = {}  # name -> versions
        self.active_versions: Dict[str, str] = {}  # name -> active_version_id
        
    def register_prompt(
        self,
        name: str,
        template: str,
        created_by: str,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> PromptVersion:
        # Generate version ID
        content_hash = hashlib.md5(template.encode()).hexdigest()[:8]
        version_num = len(self.prompts.get(name, [])) + 1
        version_id = f"{name}_v{version_num}_{content_hash}"
        
        prompt = PromptVersion(
            id=version_id,
            name=name,
            template=template,
            version=f"v{version_num}",
            created_at=datetime.now(),
            created_by=created_by,
            metadata=metadata or {"tags": tags or []},
            performance_metrics={}
        )
        
        if name not in self.prompts:
            self.prompts[name] = []
            self.active_versions[name] = version_id
        
        self.prompts[name].append(prompt)
        print(f"📝 Registered prompt {name} ({prompt.version})")
        return prompt
    
    def activate_version(self, name: str, version_id: str):
        """Switch to a different prompt version"""
        if name not in self.prompts:
            raise ValueError(f"Prompt {name} not found")
        
        valid_ids = [p.id for p in self.prompts[name]]
        if version_id not in valid_ids:
            raise ValueError(f"Version {version_id} not found for {name}")
        
        old_version = self.active_versions.get(name)
        self.active_versions[name] = version_id
        print(f"🔄 Activated {version_id} (was {old_version})")
    
    def get_active_prompt(self, name: str, **kwargs) -> str:
        """Get the active prompt template with variables filled in"""
        if name not in self.prompts:
            raise ValueError(f"Prompt {name} not found")
        
        active_id = self.active_versions[name]
        prompt = next(p for p in self.prompts[name] if p.id == active_id)
        
        # Simple template substitution
        template = prompt.template
        for key, value in kwargs.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        
        return template
    
    def ab_test(self, name: str, variant_a: str, variant_b: str, traffic_split: float = 0.5):
        """Set up A/B test between two prompt versions"""
        print(f"🧪 A/B Test: {variant_a} vs {variant_b} ({traffic_split*100:.0f}%/{(1-traffic_split)*100:.0f}%)")
        # In production, this would integrate with your routing logic
    
    def get_performance_report(self, name: str) -> Dict:
        """Compare performance across versions"""
        if name not in self.prompts:
            return {}
        
        report = {"versions": []}
        for prompt in self.prompts[name]:
            report["versions"].append({
                "version": prompt.version,
                "id": prompt.id,
                "active": prompt.id == self.active_versions.get(name),
                "metrics": prompt.performance_metrics
            })
        
        return report

# Usage
registry = PromptRegistry()

# Register initial prompt
registry.register_prompt(
    name="customer_support",
    template="""You are a helpful customer support agent.
Context: {context}
Question: {question}
Please provide a concise, accurate answer.""",
    created_by="engineer_1",
    tags=["support", "v1"]
)

# Register improved version
registry.register_prompt(
    name="customer_support",
    template="""You are an expert customer support specialist with 10 years experience.
Context: {context}
Question: {question}
Instructions:
1. Acknowledge the customer's concern
2. Provide clear, actionable steps
3. Offer additional help
Answer:""",
    created_by="engineer_2",
    tags=["support", "v2", "improved"]
)

# Get active prompt
prompt = registry.get_active_prompt(
    "customer_support",
    context="Product arrived damaged",
    question="How do I get a replacement?"
)
print("\n📄 Active Prompt:")
print(prompt)
```

### Step 5: Feedback Loop Pipeline

```python
# observability/feedback_loop.py
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class FeedbackItem:
    trace_id: str
    timestamp: datetime
    rating: int  # 1-5
    feedback_type: str  # thumbs_up, thumbs_down, correction, flag
    user_comment: Optional[str]
    corrected_output: Optional[str]  # User-provided better answer
    metadata: Dict

class FeedbackProcessor:
    def __init__(self):
        self.feedback_items: List[FeedbackItem] = []
        self.improvement_candidates: List[Dict] = []
        
    def add_feedback(self, feedback: FeedbackItem):
        self.feedback_items.append(feedback)
        
        # Auto-flag for review if low rating or correction provided
        if feedback.rating <= 2 or feedback.corrected_output:
            self.improvement_candidates.append({
                "trace_id": feedback.trace_id,
                "reason": "low_rating" if feedback.rating <= 2 else "user_correction",
                "priority": "high" if feedback.corrected_output else "medium",
                "timestamp": feedback.timestamp
            })
            print(f"🚩 Flagged {feedback.trace_id} for review")
    
    def get_improvement_insights(self) -> Dict:
        """Analyze feedback to find improvement opportunities"""
        if not self.feedback_items:
            return {}
        
        # Group by common patterns
        low_rated = [f for f in self.feedback_items if f.rating <= 2]
        corrections = [f for f in self.feedback_items if f.corrected_output]
        
        insights = {
            "total_feedback": len(self.feedback_items),
            "avg_rating": sum(f.rating for f in self.feedback_items) / len(self.feedback_items),
            "low_rating_count": len(low_rated),
            "correction_count": len(corrections),
            "common_issues": [],
            "suggested_actions": []
        }
        
        # Analyze corrections for patterns
        if corrections:
            insights["suggested_actions"].append(
                f"Review {len(corrections)} user corrections for fine-tuning data"
            )
        
        # Check for specific issue patterns
        if len(low_rated) > len(self.feedback_items) * 0.2:
            insights["common_issues"].append("High rate of low ratings (>20%)")
            insights["suggested_actions"].append("Conduct error analysis on low-rated responses")
        
        return insights
    
    def generate_finetuning_dataset(self, min_rating: int = 4) -> List[Dict]:
        """Create fine-tuning dataset from high-rated examples"""
        dataset = []
        
        for feedback in self.feedback_items:
            if feedback.rating >= min_rating:
                # Would need to fetch original input from trace
                dataset.append({
                    "trace_id": feedback.trace_id,
                    # "input": ...,  # Fetch from trace storage
                    # "output": ..., # Fetch from trace storage
                    "rating": feedback.rating
                })
        
        print(f"📊 Generated {len(dataset)} high-quality examples for fine-tuning")
        return dataset
    
    def export_feedback_report(self, filepath: str = "feedback_report.json"):
        """Export comprehensive feedback report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_improvement_insights(),
            "recent_feedback": [
                {
                    "trace_id": f.trace_id,
                    "rating": f.rating,
                    "type": f.feedback_type,
                    "has_correction": bool(f.corrected_output)
                }
                for f in self.feedback_items[-20:]  # Last 20
            ],
            "improvement_candidates": self.improvement_candidates
        }
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Exported feedback report to {filepath}")
        return report

# Usage
processor = FeedbackProcessor()

# Simulate incoming feedback
processor.add_feedback(FeedbackItem(
    trace_id="trace_123",
    timestamp=datetime.now(),
    rating=5,
    feedback_type="thumbs_up",
    user_comment="Very helpful!",
    corrected_output=None,
    metadata={"user_id": "user_456"}
))

processor.add_feedback(FeedbackItem(
    trace_id="trace_124",
    timestamp=datetime.now(),
    rating=2,
    feedback_type="thumbs_down",
    user_comment="Answer was too vague",
    corrected_output=None,
    metadata={"user_id": "user_789"}
))

processor.add_feedback(FeedbackItem(
    trace_id="trace_125",
    timestamp=datetime.now(),
    rating=1,
    feedback_type="correction",
    user_comment="This is wrong",
    corrected_output="Here's the correct answer: ...",
    metadata={"user_id": "user_101"}
))

# Get insights
insights = processor.get_improvement_insights()
print("\n📈 Feedback Insights:")
print(f"Average Rating: {insights['avg_rating']:.2f}/5")
print(f"Low Ratings: {insights['low_rating_count']}")
print(f"User Corrections: {insights['correction_count']}")
print(f"Suggested Actions: {insights['suggested_actions']}")
```

---

## 🚀 Production Deployment Checklist

### Infrastructure Setup
- [ ] Set up tracing backend (LangSmith/Phoenix/MLflow)
- [ ] Configure Prometheus + Grafana dashboards
- [ ] Set up alerting rules (Slack/PagerDuty integration)
- [ ] Create cost budgets per team/project
- [ ] Implement log aggregation (ELK stack or equivalent)

### Monitoring Implementation
- [ ] Instrument all LLM calls with tracing
- [ ] Track token usage and costs in real-time
- [ ] Monitor latency percentiles (p50, p95, p99)
- [ ] Set up quality metric collection pipeline
- [ ] Implement drift detection alerts

### Continuous Improvement
- [ ] Collect user feedback (thumbs up/down, ratings)
- [ ] Create weekly quality reports
- [ ] Maintain prompt version registry
- [ ] Run regular A/B tests on prompts
- [ ] Build fine-tuning dataset from feedback

### Security & Compliance
- [ ] Redact PII from logs and traces
- [ ] Implement access controls for trace data
- [ ] Set up audit logging for admin actions
- [ ] Ensure GDPR compliance for user data
- [ ] Monitor for prompt injection attempts

---

## 📊 Sample Dashboard Metrics

Create a Grafana dashboard with these panels:

1. **Real-time Overview**
   - Requests per minute
   - Average latency (p50, p95)
   - Error rate
   - Active users

2. **Cost Analysis**
   - Daily cost trend
   - Cost by model
   - Cost per user/team
   - Budget utilization %

3. **Quality Metrics**
   - Average user rating (rolling 7-day)
   - Hallucination rate trend
   - Toxicity incidents
   - Feedback volume

4. **Model Performance**
   - Token usage distribution
   - Cache hit rate
   - Model fallback frequency
   - Rate limit errors

5. **Drift Detection**
   - Quality metric anomalies
   - Response length distribution
   - Topic distribution shift
   - Sentiment trend

---

## 🔗 Integration with Previous Modules

This module connects to:
- **Module 04 (Evaluation)**: Continuous evaluation in production
- **Module 05 (Deployment)**: Monitoring deployed services
- **Module 06 (Optimization)**: Track optimization impact
- **Module 07 (Agents)**: Trace multi-agent workflows

---

## 🎯 Next Steps

1. **Start Small**: Begin with basic tracing and cost tracking
2. **Iterate**: Add quality metrics as you define them
3. **Automate**: Set up alerts before you need them
4. **Learn**: Use feedback to improve prompts and models
5. **Scale**: Build dashboards for stakeholders

## 📚 Recommended Tools

| Category | Open Source | Commercial |
|----------|-------------|------------|
| **Tracing** | Arize Phoenix, MLflow | LangSmith, Weights & Biases |
| **Monitoring** | Prometheus + Grafana | Datadog, New Relic |
| **Evaluation** | Ragas, TruLens | Arthur AI, Fiddler |
| **Feature Store** | Feast | Tecton, Databricks |

---

**Remember**: Observability is not optional in production LLM systems. You can't improve what you can't measure!
