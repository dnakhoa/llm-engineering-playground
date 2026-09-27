import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

// Current model IDs from llm/models.json. Neither call sends a temperature: both of
// these models reject a non-default one (they use effort instead).

// OpenAI — the Responses API
const openai = new OpenAI();

async function chatOpenAI(prompt: string): Promise<string> {
  const response = await openai.responses.create({
    model: "gpt-6-luna",
    input: prompt,
  });
  return response.output_text;
}

// Anthropic — the Messages API
const anthropic = new Anthropic();

async function chatAnthropic(prompt: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-5",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  const textBlock = response.content.find((b) => b.type === "text");
  return textBlock?.type === "text" ? textBlock.text : "";
}

// Usage
async function main() {
  console.log("OpenAI:", await chatOpenAI("What is 2+2?"));
  console.log("Anthropic:", await chatAnthropic("What is 2+2?"));
}

main();
