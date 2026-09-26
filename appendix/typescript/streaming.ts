import OpenAI from "openai";

const openai = new OpenAI();

async function streamChat(prompt: string): Promise<void> {
  // The Responses API streams typed events; text arrives as output_text deltas.
  const stream = await openai.responses.create({
    model: "gpt-6-luna", // a current OpenAI model from llm/models.json
    input: prompt,
    stream: true,
  });

  process.stdout.write("Response: ");
  for await (const event of stream) {
    if (event.type === "response.output_text.delta") {
      process.stdout.write(event.delta);
    }
  }
  console.log(); // newline
}

// Usage
async function main() {
  await streamChat("Explain quantum computing in 3 sentences.");
}

main();
