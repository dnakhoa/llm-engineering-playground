# The Flagship Agent is built from scratch, without an agent framework

The Flagship Agent is a small loop written directly against the provider SDKs, so a Reader can read it in one sitting. Frameworks (LangGraph, OpenAI Agents SDK, Google ADK 2.0) are covered in the Appendix. Framework churn is what made Modules 02, 07 and 11 stale (pre-1.0 LangChain imports). A readable loop is also what the 2026 breakout teaching repos are starred for. And the provider-agnostic decision (ADR 0002) is easier to keep without a framework's own model abstraction in the way.
