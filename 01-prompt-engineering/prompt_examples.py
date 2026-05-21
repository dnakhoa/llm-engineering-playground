"""
Prompt Engineering Examples
============================
This file demonstrates various prompting techniques you can use with LLMs.

Note: To run these examples, you'll need an API key from OpenAI, Anthropic, or another LLM provider.
You can also adapt these for local models using Ollama, vLLM, or Hugging Face.
"""

# Example 1: Zero-Shot Prompting
zero_shot_prompt = """
Translate the following text to Spanish:

Text: "The quick brown fox jumps over the lazy dog."

Translation:
"""

# Example 2: Few-Shot Prompting
few_shot_prompt = """
Convert these informal messages to professional business language:

Example 1:
Input: "Hey, can we meet tomorrow?"
Output: "Would it be possible to schedule a meeting for tomorrow?"

Example 2:
Input: "I need this done ASAP"
Output: "Could you please prioritize this task at your earliest convenience?"

Example 3:
Input: "What's the status on that project?"
Output: 
"""

# Example 3: Chain-of-Thought Prompting
cot_prompt = """
Solve this problem step by step, showing your reasoning:

A store sells apples for $2 each and oranges for $3 each. 
If Sarah buys 4 apples and 3 oranges, and pays with a $20 bill, 
how much change will she receive?

Let's think through this systematically:
"""

# Example 4: Role Prompting
role_prompt = """
You are a senior software architect with expertise in system design and best practices.

Your task is to review the following code snippet and provide:
1. Potential security vulnerabilities
2. Performance optimization suggestions
3. Code quality improvements
4. Alternative approaches if applicable

Code:
```python
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = database.execute(query)
    return result
```

Please provide your expert analysis:
"""

# Example 5: Structured Output Prompting
structured_output_prompt = """
Extract information from the following job posting and return it as a JSON object.

Required fields:
- job_title (string)
- company_name (string)
- location (string)
- salary_range (string or null)
- required_skills (array of strings)
- experience_level (string)

Job Posting:
"We are seeking a Senior Python Developer to join our team at TechCorp Inc. 
The position is based in San Francisco, CA (remote options available). 
Salary range: $120,000 - $160,000 depending on experience. 
Must have 5+ years of experience with Python, Django, PostgreSQL, and AWS.
Experience with Docker and Kubernetes is a plus."

JSON Output:
"""

# Example 6: Template-Based Prompting
def create_summary_prompt(text, max_length=100, tone="professional"):
    """Create a customizable summary prompt template."""
    return f"""
    Please summarize the following text with these requirements:
    - Maximum length: {max_length} words
    - Tone: {tone}
    - Include key points only
    - Use bullet points
    
    Text to summarize:
    \"{text}\"
    
    Summary:
    """

# Example 7: Comparison Prompting
comparison_prompt = """
Compare and contrast the following two approaches to error handling in Python:

Approach A: Using try-except blocks
```python
try:
    result = risky_operation()
except SpecificError as e:
    handle_error(e)
```

Approach B: Using result types / Either pattern
```python
result = risky_operation()
if result.is_error():
    handle_error(result.error)
else:
    use_result(result.value)
```

Provide a comparison table covering:
- Readability
- Performance
- Error visibility
- Best use cases
- Trade-offs
"""

# Example 8: Iterative Refinement Prompt
iterative_prompt = """
I'll give you a draft email. Please improve it in three iterations:

Draft: "hey, just checking if you got my last email about the thing we discussed. let me know thx"

Iteration 1: Make it professional
Iteration 2: Add more context and clarity
Iteration 3: Optimize for action and response

Show all three versions:
"""


if __name__ == "__main__":
    print("Prompt Engineering Examples")
    print("=" * 50)
    print("\nThis file contains various prompt templates.")
    print("To use them:")
    print("1. Copy any prompt variable")
    print("2. Send it to your LLM of choice (via API or UI)")
    print("3. Observe and iterate on the results")
    print("\nTip: Small changes in wording can significantly affect output quality!")
