# Memory & Context Management for LLM Systems

## 🎯 Learning Objectives
- Implement short-term and long-term memory for agents
- Manage conversation context efficiently
- Build hierarchical memory systems
- Optimize token usage with smart context windowing
- Create persistent user profiles
- Handle multi-session conversations

## 📚 Why Memory Matters

LLMs have no inherent memory between calls. For production applications, you need:

1. **Conversation Continuity**: Remember what was said earlier
2. **User Personalization**: Learn preferences over time
3. **Context Efficiency**: Fit relevant info in limited tokens
4. **Multi-Session Support**: Resume conversations days later
5. **Knowledge Accumulation**: Build on past interactions

## 🏗️ Memory Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Memory Manager                      │
├──────────────┬──────────────────┬───────────────────┤
│ Short-Term   │   Long-Term      │   Working         │
│ (Buffer)     │   (Vector Store) │   (Context)       │
├──────────────┼──────────────────┼───────────────────┤
│ • Last N     │ • User profiles  │ • Current task    │
│   messages   │ • Past sessions  │ • Relevant docs   │
│ • Session    │ • Learned facts  │ • Tool results    │
│   state      │ • Preferences    │ • Intermediate    │
│              │ • Relationships  │   results         │
└──────────────┴──────────────────┴───────────────────┘
```

## 🔧 Memory Types & Implementation

### 1. Short-Term Memory (Conversation Buffer)

```python
# memory/short_term.py
from typing import List, Dict, Optional
from collections import deque
import tiktoken

class ConversationBuffer:
    """Maintains recent conversation history."""
    
    def __init__(
        self, 
        max_messages: int = 20,
        max_tokens: int = 4000,
        model: str = "gpt-4"
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.buffer = deque(maxlen=max_messages * 2)  # pairs of user/assistant
        self.encoder = tiktoken.encoding_for_model(model)
    
    def add_message(self, role: str, content: str):
        """Add a message to the buffer."""
        self.buffer.append({"role": role, "content": content})
        
        # Trim if exceeds token limit
        while self._count_tokens() > self.max_tokens:
            # Remove oldest pair (user + assistant)
            if len(self.buffer) >= 2:
                self.buffer.popleft()
                self.buffer.popleft()
            else:
                self.buffer.popleft()
                break
    
    def get_messages(self) -> List[dict]:
        """Get all messages in buffer."""
        return list(self.buffer)
    
    def get_summary_context(self) -> str:
        """Get condensed summary of conversation."""
        if len(self.buffer) < 4:
            return "\n".join([m['content'] for m in self.buffer])
        
        # Keep first 2 and last N-2 messages
        recent = list(self.buffer)[-self.max_messages + 2:]
        early = list(self.buffer)[:2]
        
        return "\n".join([m['content'] for m in early + recent])
    
    def _count_tokens(self) -> int:
        """Count tokens in buffer."""
        total = 0
        for msg in self.buffer:
            total += len(self.encoder.encode(msg['content']))
        return total
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class SlidingWindowMemory(ConversationBuffer):
    """Implements sliding window with overlap."""
    
    def __init__(
        self,
        window_size: int = 10,
        overlap: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.overlap = overlap
    
    def get_window(self) -> List[dict]:
        """Get current window with overlap from previous."""
        messages = list(self.buffer)
        
        if len(messages) <= self.window_size:
            return messages
        
        # Keep overlap from previous window
        start_idx = max(0, len(messages) - self.window_size - self.overlap)
        return messages[start_idx:]
```

### 2. Long-Term Memory (Vector Store)

```python
# memory/long_term.py
from typing import List, Dict, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import hashlib
from datetime import datetime

class VectorMemory:
    """Persistent long-term memory using vector embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "./memory_store",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        self.store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.memory_index = 0
    
    def store_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        memory_type: str = "fact"
    ) -> str:
        """Store a memory with metadata."""
        doc_id = hashlib.md5(
            f"{content}{datetime.now()}".encode()
        ).hexdigest()
        
        doc_metadata = {
            **(metadata or {}),
            'id': doc_id,
            'type': memory_type,
            'timestamp': datetime.now().isoformat(),
            'accessed_count': 0,
            'last_accessed': None
        }
        
        doc = Document(
            page_content=content,
            metadata=doc_metadata
        )
        
        self.store.add_documents([doc], ids=[doc_id])
        self.memory_index += 1
        
        return doc_id
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant memories."""
        search_kwargs = {"k": k}
        if filter_type:
            search_kwargs["filter"] = {"type": filter_type}
        
        results = self.store.similarity_search_with_score(
            query,
            k=k,
            **search_kwargs
        )
        
        memories = []
        for doc, score in results:
            # Update access stats
            doc.metadata['accessed_count'] += 1
            doc.metadata['last_accessed'] = datetime.now().isoformat()
            
            memories.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': 1 / (1 + score)  # Convert distance to similarity
            })
        
        return memories
    
    def update_memory(self, memory_id: str, content: str):
        """Update existing memory."""
        # Delete old and add new
        self.store.delete(ids=[memory_id])
        return self.store_memory(content, {'updated_from': memory_id})
    
    def forget_memories(
        self,
        older_than_days: int = 90,
        min_access_count: int = 0
    ) -> int:
        """Remove old, unused memories."""
        cutoff = datetime.now().timestamp() - (older_than_days * 86400)
        
        # Get all memories
        all_docs = self.store.get()
        
        to_delete = []
        for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']):
            ts = datetime.fromisoformat(metadata.get('timestamp', '')).timestamp()
            accessed = metadata.get('accessed_count', 0)
            
            if ts < cutoff and accessed <= min_access_count:
                to_delete.append(doc_id)
        
        if to_delete:
            self.store.delete(ids=to_delete)
        
        return len(to_delete)
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Build user profile from stored memories."""
        profile_memories = self.retrieve_memories(
            f"user preferences for {user_id}",
            k=20,
            filter_type="preference"
        )
        
        fact_memories = self.retrieve_memories(
            f"facts about user {user_id}",
            k=20,
            filter_type="fact"
        )
        
        return {
            'user_id': user_id,
            'preferences': [m['content'] for m in profile_memories],
            'known_facts': [m['content'] for m in fact_memories],
            'total_memories': len(profile_memories) + len(fact_memories)
        }
```

### 3. Working Memory (Context Assembly)

```python
# memory/working.py
from typing import List, Dict, Optional, TypedDict
from dataclasses import dataclass, field
import json

@dataclass
class MemoryContext:
    """Assembled context for LLM call."""
    system_prompt: str = ""
    conversation_history: List[dict] = field(default_factory=list)
    retrieved_memories: List[str] = field(default_factory=list)
    task_context: Dict = field(default_factory=dict)
    tool_results: List[str] = field(default_factory=list)
    
    def to_messages(self) -> List[dict]:
        """Convert to message format for LLM."""
        messages = []
        
        # System prompt with memories
        if self.system_prompt or self.retrieved_memories:
            system_parts = [self.system_prompt]
            
            if self.retrieved_memories:
                system_parts.append(
                    "\n\nRelevant memories:\n" + 
                    "\n".join(f"- {m}" for m in self.retrieved_memories)
                )
            
            messages.append({
                "role": "system",
                "content": "\n".join(system_parts)
            })
        
        # Conversation history
        messages.extend(self.conversation_history)
        
        # Task context as user message
        if self.task_context:
            messages.append({
                "role": "user",
                "content": f"Task context: {json.dumps(self.task_context)}"
            })
        
        # Tool results
        if self.tool_results:
            messages.append({
                "role": "assistant",
                "content": "Tool results:\n" + "\n".join(self.tool_results)
            })
        
        return messages
    
    def count_tokens(self, encoder) -> int:
        """Estimate token count."""
        total = 0
        for msg in self.to_messages():
            total += len(encoder.encode(msg.get('content', '')))
        return total


class ContextAssembler:
    """Intelligently assembles context within token limits."""
    
    def __init__(
        self,
        max_context_tokens: int = 8000,
        priority_order: List[str] = None
    ):
        self.max_tokens = max_context_tokens
        self.priority_order = priority_order or [
            'system_prompt',
            'recent_conversation',
            'task_context',
            'relevant_memories',
            'tool_results',
            'older_conversation'
        ]
    
    def assemble(
        self,
        system_prompt: str,
        conversation: List[dict],
        memories: List[str],
        task_context: Dict,
        tool_results: List[str],
        encoder
    ) -> MemoryContext:
        """Assemble optimal context within token limits."""
        
        # Start with high-priority items
        context = MemoryContext(
            system_prompt=system_prompt,
            task_context=task_context
        )
        
        # Add recent conversation (most important)
        recent_conv = conversation[-10:]  # Last 5 exchanges
        context.conversation_history = recent_conv
        
        current_tokens = context.count_tokens(encoder)
        
        # Add memories if space
        if current_tokens < self.max_tokens * 0.7:
            available = self.max_tokens - current_tokens
            mem_tokens = sum(len(encoder.encode(m)) for m in memories[:5])
            
            if mem_tokens < available:
                context.retrieved_memories = memories[:5]
                current_tokens += mem_tokens
        
        # Add tool results if space
        if tool_results and current_tokens < self.max_tokens * 0.9:
            context.tool_results = tool_results[:3]
        
        return context
```

## 🧠 Hierarchical Memory System

```python
# memory/hierarchical.py
from typing import Optional, List, Dict
from .short_term import ConversationBuffer
from .long_term import VectorMemory
from .working import ContextAssembler, MemoryContext
import tiktoken

class HierarchicalMemory:
    """
    Multi-level memory system:
    - L1: Working memory (immediate context)
    - L2: Short-term buffer (current session)
    - L3: Long-term storage (persistent)
    """
    
    def __init__(
        self,
        user_id: str,
        session_id: str,
        l1_max_tokens: int = 4000,
        l2_max_messages: int = 50,
        l3_persist_dir: str = "./memory_db"
    ):
        self.user_id = user_id
        self.session_id = session_id
        
        # L2: Short-term
        self.short_term = ConversationBuffer(
            max_messages=l2_max_messages,
            max_tokens=8000
        )
        
        # L3: Long-term
        self.long_term = VectorMemory(
            persist_directory=f"{l3_persist_dir}/{user_id}"
        )
        
        # Context assembler
        self.assembler = ContextAssembler(
            max_context_tokens=l1_max_tokens
        )
        
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def add_interaction(
        self,
        user_input: str,
        assistant_response: str,
        extract_memories: bool = True
    ):
        """Process and store an interaction."""
        
        # Add to short-term
        self.short_term.add_message("user", user_input)
        self.short_term.add_message("assistant", assistant_response)
        
        # Extract and store important memories
        if extract_memories:
            self._extract_and_store_memories(user_input, assistant_response)
    
    def _extract_and_store_memories(
        self,
        user_input: str,
        assistant_response: str
    ):
        """Extract key information for long-term storage."""
        # Simple heuristic extraction
        # In production, use an LLM to identify memorable content
        
        # Look for preference indicators
        preference_keywords = ['i prefer', 'i like', 'i hate', 'always', 'never']
        if any(kw in user_input.lower() for kw in preference_keywords):
            self.long_term.store_memory(
                content=user_input,
                metadata={'user_id': self.user_id},
                memory_type='preference'
            )
        
        # Look for factual statements
        fact_indicators = ['i am', 'i work', 'i live', 'my name is']
        if any(kw in user_input.lower() for kw in fact_indicators):
            self.long_term.store_memory(
                content=user_input,
                metadata={'user_id': self.user_id},
                memory_type='fact'
            )
    
    def get_context(
        self,
        current_query: str,
        system_prompt: str = "",
        task_context: Dict = None
    ) -> MemoryContext:
        """Get assembled context for next LLM call."""
        
        # Retrieve relevant long-term memories
        memories = self.long_term.retrieve_memories(
            current_query,
            k=5
        )
        memory_contents = [m['content'] for m in memories]
        
        # Get short-term conversation
        conversation = self.short_term.get_messages()
        
        # Assemble working context
        return self.assembler.assemble(
            system_prompt=system_prompt,
            conversation=conversation,
            memories=memory_contents,
            task_context=task_context or {},
            tool_results=[],
            encoder=self.encoder
        )
    
    def summarize_session(self) -> str:
        """Create summary of current session for archival."""
        messages = self.short_term.get_messages()
        
        if not messages:
            return "No conversation in this session."
        
        # Create simple summary
        # In production, use LLM to generate better summary
        return f"Session {self.session_id}: {len(messages)} messages exchanged."
    
    def end_session(self, archive: bool = True):
        """End current session and optionally archive."""
        if archive:
            summary = self.summarize_session()
            self.long_term.store_memory(
                content=summary,
                metadata={
                    'user_id': self.user_id,
                    'session_id': self.session_id,
                    'type': 'session_summary'
                },
                memory_type='session'
            )
        
        self.short_term.clear()
```

## 🔄 Advanced Patterns

### 1. Reflection-Based Memory

```python
# memory/reflection.py
from typing import List, Dict

class ReflectiveMemory:
    """Memory system with self-reflection capabilities."""
    
    def __init__(self, base_memory: HierarchicalMemory, llm):
        self.memory = base_memory
        self.llm = llm
    
    async def reflect_on_interaction(
        self,
        user_input: str,
        assistant_response: str
    ) -> List[Dict]:
        """Use LLM to identify what should be remembered."""
        
        prompt = f"""
        Analyze this conversation and extract key information worth remembering:
        
        User: {user_input}
        Assistant: {assistant_response}
        
        Identify:
        1. User preferences
        2. Important facts about the user
        3. Ongoing tasks or goals
        4. Any commitments or follow-ups
        
        Return as JSON array with type and content fields.
        """
        
        response = await self.llm.generate(prompt)
        
        try:
            import json
            extractions = json.loads(response)
            
            for item in extractions:
                self.memory.long_term.store_memory(
                    content=item['content'],
                    metadata={'user_id': self.memory.user_id},
                    memory_type=item.get('type', 'general')
                )
            
            return extractions
        except:
            return []
    
    async def periodic_review(self, interval_hours: int = 24):
        """Periodically review and consolidate memories."""
        # Retrieve recent memories
        recent = self.memory.long_term.retrieve_memories(
            "recent interactions",
            k=50
        )
        
        # Ask LLM to find patterns and consolidate
        prompt = f"""
        Review these memories and identify:
        1. Redundant memories that can be merged
        2. Contradictions that need resolution
        3. Emerging patterns or themes
        
        Memories:
        {[m['content'] for m in recent]}
        """
        
        # Process consolidation suggestions
        # ... implementation
```

### 2. Episodic vs Semantic Memory

```python
# memory/episodic_semantic.py

class DualMemorySystem:
    """Separates episodic (events) and semantic (facts) memory."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.episodic = VectorMemory(
            persist_directory=f"./memory/{user_id}/episodic"
        )
        self.semantic = VectorMemory(
            persist_directory=f"./memory/{user_id}/semantic"
        )
    
    def store_episode(
        self,
        event_description: str,
        timestamp: str,
        participants: List[str] = None
    ):
        """Store an episodic memory (specific event)."""
        self.episodic.store_memory(
            content=event_description,
            metadata={
                'timestamp': timestamp,
                'participants': participants or [],
                'type': 'episode'
            },
            memory_type='episodic'
        )
    
    def store_fact(
        self,
        fact: str,
        confidence: float = 1.0,
        source: str = None
    ):
        """Store a semantic memory (general knowledge)."""
        self.semantic.store_memory(
            content=fact,
            metadata={
                'confidence': confidence,
                'source': source,
                'type': 'fact'
            },
            memory_type='semantic'
        )
    
    def retrieve_context(
        self,
        query: str,
        include_episodes: bool = True,
        include_facts: bool = True
    ) -> List[str]:
        """Retrieve relevant context from both systems."""
        results = []
        
        if include_episodes:
            episodes = self.episodic.retrieve_memories(query, k=3)
            results.extend([f"[Event] {e['content']}" for e in episodes])
        
        if include_facts:
            facts = self.semantic.retrieve_memories(query, k=5)
            results.extend([f"[Fact] {f['content']}" for f in facts])
        
        return results
```

## 📊 Memory Optimization Strategies

### 1. Importance Scoring

```python
def calculate_memory_importance(
    memory: Dict,
    access_frequency: int,
    recency_weight: float = 0.3,
    frequency_weight: float = 0.5,
    relevance_weight: float = 0.2
) -> float:
    """Calculate importance score for memory prioritization."""
    from datetime import datetime
    
    # Recency score (0-1, newer = higher)
    age_days = (
        datetime.now() - 
        datetime.fromisoformat(memory['timestamp'])
    ).days
    recency_score = max(0, 1 - (age_days / 90))
    
    # Frequency score (0-1)
    frequency_score = min(1, access_frequency / 10)
    
    # Relevance would be calculated per-query
    
    # Weighted combination
    importance = (
        recency_weight * recency_score +
        frequency_weight * frequency_score
    )
    
    return importance
```

### 2. Memory Compression

```python
async def compress_memories(
    memories: List[str],
    llm,
    target_length: int = 500
) -> str:
    """Compress multiple memories into concise summary."""
    
    prompt = f"""
    Compress these memories into a concise summary under {target_length} characters:
    
    {chr(10).join(memories)}
    
    Preserve key facts and relationships. Remove redundancy.
    """
    
    compressed = await llm.generate(prompt)
    return compressed[:target_length]
```

## 🎯 Best Practices

1. **Tiered Storage**: Hot (RAM) → Warm (SSD) → Cold (S3)
2. **Lazy Loading**: Load memories on-demand, not upfront
3. **Forgetting Mechanism**: Remove outdated/irrelevant memories
4. **Privacy First**: Encrypt sensitive memories, allow deletion
5. **User Control**: Let users view/edit/delete their memories
6. **Versioning**: Track memory changes over time
7. **Testing**: Validate memory retrieval quality regularly

## 📝 Example Usage

```python
# examples/memory_usage.py
from memory.hierarchical import HierarchicalMemory

# Initialize
memory = HierarchicalMemory(
    user_id="user_123",
    session_id="session_456"
)

# Add interactions
memory.add_interaction(
    user_input="I prefer Python over JavaScript for backend development",
    assistant_response="That's great! Python is excellent for backend..."
)

memory.add_interaction(
    user_input="I work at a fintech company in New York",
    assistant_response="Interesting! Fintech is a growing field..."
)

# Get context for next query
context = memory.get_context(
    current_query="What programming languages do I like?",
    system_prompt="You are a helpful assistant with memory of past conversations."
)

# Use context with LLM
messages = context.to_messages()
# response = llm.chat(messages)

# End session
memory.end_session(archive=True)
```

## 🔧 Configuration

```yaml
# configs/memory_config.yaml
memory:
  short_term:
    max_messages: 50
    max_tokens: 8000
    trim_strategy: "fifo"
  
  long_term:
    persist_directory: "./data/memory"
    embedding_model: "all-MiniLM-L6-v2"
    retrieval_k: 5
    
    retention:
      max_age_days: 365
      min_access_count: 2
      review_interval_days: 30
  
  context:
    max_tokens: 8000
    priority_order:
      - system_prompt
      - recent_conversation
      - relevant_memories
      - task_context
  
  privacy:
    encrypt_at_rest: true
    auto_delete_on_request: true
    audit_logging: true
```

---

**Effective memory management is crucial for building intelligent, personalized LLM applications!**
