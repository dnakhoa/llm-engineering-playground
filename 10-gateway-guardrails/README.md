# API Gateway & Guardrails for LLM Systems

## 🎯 Learning Objectives
- Implement production API gateways for LLM services
- Build comprehensive guardrails for safety and compliance
- Handle rate limiting, authentication, and authorization
- Detect and prevent prompt injection attacks
- Implement content filtering and moderation
- Create audit trails for compliance

## 📚 Why Gateway & Guardrails Matter

In production LLM systems, you need:
1. **Security**: Protect against malicious inputs
2. **Compliance**: Meet regulatory requirements (GDPR, HIPAA)
3. **Cost Control**: Prevent abuse and manage quotas
4. **Quality**: Filter harmful or low-quality outputs
5. **Observability**: Track all interactions for auditing

## 🏗️ Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │ →→→ │   Gateway    │ →→→ │  LLM Service│
│  (App/User) │     │ + Guardrails │     │   Backend   │
└─────────────┘     └──────────────┘     └─────────────┘
                           ↓
                    ┌──────────────┐
                    │  Monitoring  │
                    │  & Logging   │
                    └──────────────┘
```

## 🔒 Gateway Components

### 1. Authentication & Authorization

```python
# gateway/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import Optional

security = HTTPBearer()

class AuthManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    async def verify_token(
        self, 
        creds: HTTPAuthorizationCredentials = Security(security)
    ) -> dict:
        """Verify JWT token and extract user claims."""
        try:
            payload = jwt.decode(
                creds.credentials,
                self.secret_key,
                algorithms=["HS256"]
            )
            return {
                "user_id": payload.get("sub"),
                "roles": payload.get("roles", []),
                "quota": payload.get("quota", 1000)
            }
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_permission(self, user_claims: dict, required_role: str) -> bool:
        """Check if user has required role."""
        return required_role in user_claims.get("roles", [])
```

### 2. Rate Limiting & Quota Management

```python
# gateway/rate_limiter.py
from redis import Redis
from datetime import datetime, timedelta
from typing import Optional
import json

class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        limit: int = 100,
        window: int = 60  # seconds
    ) -> tuple[bool, Optional[int]]:
        """
        Check if user is within rate limit.
        Returns: (allowed, retry_after_seconds)
        """
        key = f"rate_limit:{user_id}"
        current = self.redis.incr(key)
        
        if current == 1:
            self.redis.expire(key, window)
        
        if current > limit:
            ttl = self.redis.ttl(key)
            return False, ttl
        
        return True, None
    
    async def check_quota(self, user_id: str, tokens_needed: int) -> bool:
        """Check if user has remaining token quota."""
        key = f"quota:{user_id}"
        remaining = self.redis.get(key)
        
        if remaining is None:
            # Set default quota
            remaining = 10000
            self.redis.setex(key, timedelta(days=30), remaining)
        
        remaining = int(remaining)
        if remaining < tokens_needed:
            return False
        
        self.redis.decrby(key, tokens_needed)
        return True
```

### 3. Request Validation

```python
# gateway/validator.py
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re

class LLMRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=100)
    messages: List[dict]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = False
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        
        for i, msg in enumerate(v):
            if 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Message {i} missing role or content")
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role in message {i}")
            
            # Check for suspicious patterns
            content = msg.get('content', '')
            if cls._detect_prompt_injection(content):
                raise ValueError("Potential prompt injection detected")
        
        return v
    
    @staticmethod
    def _detect_prompt_injection(text: str) -> bool:
        """Detect common prompt injection patterns."""
        patterns = [
            r"ignore previous instructions",
            r"forget all.*rules",
            r"you are now.*without restrictions",
            r"output your.*system prompt",
            r"bypass.*safety",
        ]
        
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
```

## 🛡️ Guardrails Implementation

### 1. Input Guardrails

```python
# guardrails/input_filters.py
from typing import Tuple, List
import re

class InputGuardrail:
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        }
        
        self.toxic_keywords = [
            'hate speech', 'harassment', 'violence',
            'self-harm', 'suicide', 'bomb', 'weapon'
        ]
    
    def check_input(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate input text.
        Returns: (is_safe, list_of_issues)
        """
        issues = []
        
        # Check for PII
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                issues.append(f"Contains {pii_type.upper()}")
        
        # Check for toxic content
        text_lower = text.lower()
        for keyword in self.toxic_keywords:
            if keyword in text_lower:
                issues.append(f"Potentially toxic: {keyword}")
        
        # Check length
        if len(text) > 10000:
            issues.append("Input too long (>10k chars)")
        
        # Check for injection attempts
        if self._detect_injection(text):
            issues.append("Potential prompt injection")
        
        return len(issues) == 0, issues
    
    def _detect_injection(self, text: str) -> bool:
        """Advanced injection detection."""
        # Check for delimiter manipulation
        dangerous_delimiters = ['###', '"""', "'''", '```']
        if text.count('###') > 2:
            return True
        
        # Check for role-playing attempts
        injection_phrases = [
            "you are now",
            "disregard all",
            "ignore safety",
            "output everything",
            "what is your system prompt"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in injection_phrases)
    
    def sanitize_input(self, text: str) -> str:
        """Remove or mask sensitive information."""
        # Mask email addresses
        text = re.sub(
            self.pii_patterns['email'],
            '[EMAIL_REDACTED]',
            text
        )
        
        # Mask phone numbers
        text = re.sub(
            self.pii_patterns['phone'],
            '[PHONE_REDACTED]',
            text
        )
        
        return text
```

### 2. Output Guardrails

```python
# guardrails/output_filters.py
from typing import Tuple, List, Dict
import json

class OutputGuardrail:
    def __init__(self):
        self.fact_check_enabled = True
        self.json_validator = True
    
    def check_output(
        self, 
        text: str, 
        expected_format: str = None
    ) -> Tuple[bool, List[str], dict]:
        """
        Validate LLM output.
        Returns: (is_valid, issues, metadata)
        """
        issues = []
        metadata = {
            'length': len(text),
            'contains_code': '```' in text,
            'contains_urls': 'http' in text
        }
        
        # Check format if specified
        if expected_format == 'json':
            is_valid_json, json_issue = self._validate_json(text)
            if not is_valid_json:
                issues.append(json_issue)
        
        # Check for hallucination indicators
        hallucination_score = self._detect_hallucination(text)
        if hallucination_score > 0.7:
            issues.append(f"High hallucination risk: {hallucination_score:.2f}")
            metadata['hallucination_score'] = hallucination_score
        
        # Check for toxic output
        toxicity_score = self._check_toxicity(text)
        if toxicity_score > 0.5:
            issues.append(f"Toxic content detected: {toxicity_score:.2f}")
            metadata['toxicity_score'] = toxicity_score
        
        return len(issues) == 0, issues, metadata
    
    def _validate_json(self, text: str) -> Tuple[bool, str]:
        """Validate JSON output."""
        try:
            # Try to extract JSON from markdown code blocks
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                text = text[start:end].strip()
            elif '```' in text:
                start = text.find('```') + 3
                end = text.find('```', start)
                text = text[start:end].strip()
            
            json.loads(text)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    def _detect_hallucination(self, text: str) -> float:
        """Simple heuristic for hallucination detection."""
        # In production, use a dedicated model
        uncertainty_phrases = [
            "i think", "probably", "might be",
            "not sure", "could be", "perhaps"
        ]
        
        text_lower = text.lower()
        count = sum(1 for phrase in uncertainty_phrases if phrase in text_lower)
        
        # Normalize to 0-1 score
        return min(count / 5.0, 1.0)
    
    def _check_toxicity(self, text: str) -> float:
        """Simple toxicity check."""
        # In production, use Perspective API or similar
        toxic_words = ['stupid', 'idiot', 'hate', 'kill', 'die']
        text_lower = text.lower()
        
        count = sum(1 for word in toxic_words if word in text_lower)
        return min(count / 10.0, 1.0)
```

### 3. Compliance Guardrails

```python
# guardrails/compliance.py
from datetime import datetime
from typing import Dict, List
import hashlib
import json

class ComplianceLogger:
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.retention_days = 365
    
    def log_interaction(
        self,
        user_id: str,
        request: dict,
        response: dict,
        guardrail_results: dict,
        metadata: dict
    ) -> str:
        """Log interaction for compliance auditing."""
        interaction_id = self._generate_id(user_id, request)
        
        record = {
            'interaction_id': interaction_id,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'request_hash': hashlib.sha256(
                json.dumps(request).encode()
            ).hexdigest(),
            'response_hash': hashlib.sha256(
                json.dumps(response).encode()
            ).hexdigest(),
            'guardrail_passed': guardrail_results.get('passed', False),
            'guardrail_issues': guardrail_results.get('issues', []),
            'metadata': metadata,
            'retention_until': (
                datetime.utcnow().replace(
                    year=datetime.utcnow().year + 1
                )
            ).isoformat()
        }
        
        # Store in compliant storage (e.g., S3 with encryption)
        self.storage.save(record)
        
        return interaction_id
    
    def get_audit_trail(
        self, 
        user_id: str, 
        start_date: datetime,
        end_date: datetime
    ) -> List[dict]:
        """Retrieve audit trail for compliance review."""
        return self.storage.query(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
```

## 🚀 Complete Gateway Example

```python
# examples/gateway_server.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

from gateway.auth import AuthManager
from gateway.rate_limiter import RateLimiter
from gateway.validator import LLMRequest
from guardrails.input_filters import InputGuardrail
from guardrails.output_filters import OutputGuardrail
from guardrails.compliance import ComplianceLogger

app = FastAPI(title="LLM Gateway")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
auth_manager = AuthManager(secret_key="your-secret-key")
rate_limiter = RateLimiter(redis_client=None)  # Configure Redis
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()
compliance_logger = ComplianceLogger(storage_backend=None)

@app.post("/v1/chat/completions")
async def chat_completions(
    request: LLMRequest,
    user_claims: dict = Depends(auth_manager.verify_token)
):
    """Protected LLM endpoint with full guardrails."""
    start_time = time.time()
    
    # Check rate limit
    allowed, retry_after = await rate_limiter.check_rate_limit(
        user_claims['user_id']
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            detail="Rate limit exceeded"
        )
    
    # Check quota
    estimated_tokens = sum(len(m['content']) // 4 for m in request.messages)
    if not await rate_limiter.check_quota(user_claims['user_id'], estimated_tokens):
        raise HTTPException(status_code=403, detail="Quota exceeded")
    
    # Input guardrails
    full_text = "\n".join([m['content'] for m in request.messages])
    is_safe, issues = input_guardrail.check_input(full_text)
    
    if not is_safe:
        return {
            "error": "Content policy violation",
            "details": issues,
            "blocked": True
        }
    
    # Call LLM service (placeholder)
    # response = await call_llm_service(request)
    response = {"choices": [{"message": {"content": "Response"}}]}
    
    # Output guardrails
    output_text = response['choices'][0]['message']['content']
    is_valid, out_issues, out_metadata = output_guardrail.check_output(
        output_text,
        expected_format=None
    )
    
    if not is_valid:
        # Handle invalid output (retry, fallback, or block)
        response['guardrail_warnings'] = out_issues
    
    # Log for compliance
    interaction_id = compliance_logger.log_interaction(
        user_id=user_claims['user_id'],
        request=request.dict(),
        response=response,
        guardrail_results={'passed': is_safe and is_valid, 'issues': issues + out_issues},
        metadata={
            'latency_ms': (time.time() - start_time) * 1000,
            'model': request.model,
            'tokens_estimated': estimated_tokens
        }
    )
    
    response['interaction_id'] = interaction_id
    response['metadata'] = out_metadata
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 📊 Monitoring & Alerting

```python
# gateway/monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['endpoint', 'status', 'model']
)

REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'LLM request latency',
    ['endpoint', 'model']
)

GUARDRAIL_BLOCKS = Counter(
    'llm_guardrail_blocks_total',
    'Requests blocked by guardrails',
    ['reason']
)

TOKEN_USAGE = Gauge(
    'llm_tokens_remaining',
    'Remaining token quota',
    ['user_id']
)

class GatewayMonitor:
    def __init__(self):
        pass
    
    def track_request(self, endpoint: str, model: str, status: str):
        REQUEST_COUNT.labels(endpoint=endpoint, status=status, model=model).inc()
    
    def track_latency(self, endpoint: str, model: str, duration: float):
        REQUEST_LATENCY.labels(endpoint=endpoint, model=model).observe(duration)
    
    def track_block(self, reason: str):
        GUARDRAIL_BLOCKS.labels(reason=reason).inc()
    
    def update_quota(self, user_id: str, remaining: int):
        TOKEN_USAGE.labels(user_id=user_id).set(remaining)
```

## 🔧 Configuration

```yaml
# configs/gateway_config.yaml
gateway:
  port: 8000
  host: "0.0.0.0"
  cors_origins: ["https://your-app.com"]

auth:
  jwt_secret: "${JWT_SECRET}"
  token_expiry: 3600
  
rate_limits:
  default:
    requests_per_minute: 60
    tokens_per_day: 100000
  premium:
    requests_per_minute: 300
    tokens_per_day: 1000000

guardrails:
  input:
    max_length: 10000
    detect_pii: true
    detect_injection: true
    block_toxic: true
  output:
    validate_json: true
    check_hallucination: true
    max_toxicity: 0.5

compliance:
  log_all_interactions: true
  retention_days: 365
  encrypt_at_rest: true

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  alerting:
    high_error_rate: 0.05
    high_latency_p99: 5.0
```

## 🎯 Best Practices

1. **Defense in Depth**: Multiple layers of security
2. **Fail Secure**: Block on uncertainty
3. **Audit Everything**: Complete interaction logs
4. **Regular Updates**: Keep injection patterns current
5. **Human Review**: Escalate edge cases
6. **Performance**: Cache validation results
7. **Transparency**: Inform users about blocks

## 📝 Next Steps

1. Deploy gateway with your LLM service
2. Customize guardrails for your domain
3. Set up monitoring dashboards
4. Configure alerting rules
5. Regular security audits

## 🧪 Hands-On Exercises

1. **Injection Attack Suite**: Write 10 different prompt injection techniques (delimiter manipulation, role-playing, instruction override, encoding tricks). Test them against the `InputGuardrail`. How many get through?

2. **PII Redaction**: Build a redactor that handles emails, phone numbers, SSNs, credit cards, and street addresses. Test it on a paragraph containing all 5 types. Does it miss anything?

3. **Rate Limiter Under Load**: Simulate 200 requests from 10 users with a limit of 20 requests/minute per user. Track which requests get blocked and verify the counts are correct.

4. **Output Hallucination Detector**: Build a simple fact-checker that compares LLM output against a list of known facts. How many hallucinated claims does it catch on 20 test outputs?

5. **Full Gateway Integration**: Wire up auth, rate limiting, input guardrails, and output guardrails into a single FastAPI middleware chain. Test it end-to-end with valid requests, injection attempts, and quota-exceeded scenarios.

---

**Production-ready LLM systems need robust gateways and guardrails!**
