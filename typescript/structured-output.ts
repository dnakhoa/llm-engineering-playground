import OpenAI from "openai";
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
  const response = await openai.beta.chat.completions.parse({
    model: "gpt-4o-mini",
    response_format: zodResponseFormat(PersonSchema, "person"),
    messages: [
      { role: "user", content: `Extract person info from: ${text}` },
    ],
  });

  return response.choices[0].message.parsed!;
}

// Usage
async function main() {
  const person = await extractPerson("John Smith is a 35-year-old engineer.");
  console.log(person); // { name: "John Smith", age: 35, occupation: "engineer" }
}

main();
