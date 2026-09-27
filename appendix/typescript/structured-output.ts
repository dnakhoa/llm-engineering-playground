import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const openai = new OpenAI();

// Define schema with Zod
const PersonSchema = z.object({
  name: z.string(),
  age: z.number(),
  occupation: z.string(),
});

type Person = z.infer<typeof PersonSchema>;

async function extractPerson(text: string): Promise<Person> {
  // Structured output on the Responses API: the schema goes in text.format.
  const response = await openai.responses.parse({
    model: "gpt-6-luna", // a current OpenAI model from llm/models.json
    input: [{ role: "user", content: `Extract person info from: ${text}` }],
    text: { format: zodTextFormat(PersonSchema, "person") },
  });

  if (!response.output_parsed) {
    throw new Error("The model returned no parsable person.");
  }
  return response.output_parsed;
}

// Usage
async function main() {
  const person = await extractPerson("John Smith is a 35-year-old engineer.");
  console.log(person); // { name: "John Smith", age: 35, occupation: "engineer" }
}

main();
