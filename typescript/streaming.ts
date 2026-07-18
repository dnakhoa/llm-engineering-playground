import OpenAI from "openai";

const openai = new OpenAI();

async function streamChat(prompt: string): Promise<void> {
  const stream = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: prompt }],
    stream: true,
  });

  process.stdout.write("Response: ");
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
    }
  }
  console.log(); // newline
}

// Usage
async function main() {
  await streamChat("Explain quantum computing in 3 sentences.");
}

main();
