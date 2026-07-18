import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

// OpenAI
const openai = new OpenAI();

async function chatOpenAI(prompt: string): Promise<string> {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: prompt }],
    temperature: 0.7,
  });
  return response.choices[0].message.content ?? "";
}

// Anthropic
const anthropic = new Anthropic();

async function chatAnthropic(prompt: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-5",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });
  const textBlock = response.content.find((b) => b.type === "text");
  return textBlock?.text ?? "";
}

// Usage
async function main() {
  console.log("OpenAI:", await chatOpenAI("What is 2+2?"));
  console.log("Anthropic:", await chatAnthropic("What is 2+2?"));
}

main();
