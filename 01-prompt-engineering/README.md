# Module 1: Prompt Engineering

## What is Prompt Engineering?

Prompt engineering is the art and science of designing effective prompts to guide Large Language Models (LLMs) to produce desired outputs. It's the foundation of working with LLMs without modifying their weights.

## Key Concepts

### 1. Zero-Shot Prompting
Asking the model to perform a task without any examples.

**Example:**
```
Translate this to French: "Hello, how are you?"
```

### 2. Few-Shot Prompting
Providing examples in the prompt to guide the model's behavior.

**Example:**
```
Convert to formal tone:
Input: "Hey, what's up?"
Output: "Greetings, how are you doing?"

Input: "Thanks a lot!"
Output: "Thank you very much!"

Input: "Can you help me?"
Output: 
```

### 3. Chain-of-Thought (CoT)
Encouraging the model to show its reasoning process before giving the final answer.

**Example:**
```
Q: John has 5 apples. He buys 3 more, then gives away 2. How many does he have?
A: Let's think step by step:
- John starts with 5 apples
- He buys 3 more: 5 + 3 = 8 apples
- He gives away 2: 8 - 2 = 6 apples
- Final answer: 6 apples
```

### 4. Role Prompting
Assigning a specific role or persona to the model.

**Example:**
```
You are an expert Python developer with 10 years of experience. Review the following code and suggest improvements...
```

### 5. Structured Output
Requesting output in a specific format (JSON, XML, Markdown, etc.)

**Example:**
```
Extract the following information from the text and return as JSON:
- Name
- Age
- Occupation

Text: "John Smith is a 35-year-old software engineer..."
```

## Best Practices

1. **Be Clear and Specific**: Vague prompts lead to vague answers
2. **Provide Context**: Give relevant background information
3. **Use Delimiters**: Separate instructions from content using ```, """, or ---
4. **Specify Format**: Tell the model exactly how you want the output
5. **Iterate**: Refine prompts based on results

## Hands-On Exercises

Two ways to learn hands-on:

1. **Interactive notebook** (recommended): Open `prompt_engineering.ipynb` and run cells one at a time — you'll see LLM responses inline.
2. **Script**: Run `python prompt_examples.py` to execute all 8 examples end-to-end.

**Challenges to try:**
- Modify the few-shot examples in Example 2 and observe how output style changes
- Add a `role` parameter to Example 4 and compare responses (e.g., "pirate", "professor")
- Change the JSON schema in Example 5 to extract different fields
- Build your own chain-of-thought prompt for a math problem

## Next Steps

After mastering prompt engineering, move to Module 2: RAG Systems, where you'll learn how to augment LLMs with external knowledge.
