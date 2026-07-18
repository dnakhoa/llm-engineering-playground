import OpenAI from "openai";

const openai = new OpenAI();

// Simple in-memory vector store
interface Doc {
  text: string;
  embedding: number[];
}

const docs: Doc[] = [];

async function embed(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

async function addDocument(text: string): Promise<void> {
  docs.push({ text, embedding: await embed(text) });
}

async function retrieve(query: string, topK = 3): Promise<string[]> {
  const queryEmbedding = await embed(query);
  const scored = docs.map((doc) => ({
    text: doc.text,
    score: cosineSimilarity(queryEmbedding, doc.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((d) => d.text);
}

async function ragQuery(query: string): Promise<string> {
  const context = await retrieve(query);
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "system",
        content: `Answer based on this context:\n${context.join("\n\n")}`,
      },
      { role: "user", content: query },
    ],
  });
  return response.choices[0].message.content ?? "";
}

// Usage
async function main() {
  await addDocument("Python is great for data science and ML.");
  await addDocument("TypeScript adds types to JavaScript.");
  await addDocument("Rust is fast and memory-safe.");

  console.log(await ragQuery("What language is best for ML?"));
}

main();
