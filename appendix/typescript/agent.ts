import OpenAI from "openai";

const openai = new OpenAI();

// A current OpenAI model from llm/models.json. Current OpenAI models make tool calls
// through the Responses API; Chat Completions only does it with reasoning turned off.
const MODEL = "gpt-6-luna";
const MAX_STEPS = 10; // stop condition: never loop forever on a model that keeps calling tools

// Define tools (Responses API shape: name, description and parameters at the top level)
const tools: OpenAI.Responses.FunctionTool[] = [
  {
    type: "function",
    name: "get_weather",
    description: "Get current weather for a location",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "City name" },
      },
      required: ["location"],
      additionalProperties: false,
    },
    strict: true,
  },
  {
    type: "function",
    name: "search_docs",
    description: "Search internal documentation",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
      },
      required: ["query"],
      additionalProperties: false,
    },
    strict: true,
  },
];

// Tool implementations
function getWeather(location: string): string {
  return `Weather in ${location}: 72°F, sunny`;
}

function searchDocs(query: string): string {
  return `Found 3 docs about "${query}"`;
}

function runTool(name: string, args: Record<string, string>): string {
  if (name === "get_weather") return getWeather(args.location);
  if (name === "search_docs") return searchDocs(args.query);
  return "Unknown tool";
}

async function agentLoop(userMessage: string): Promise<string> {
  const input: OpenAI.Responses.ResponseInput = [{ role: "user", content: userMessage }];

  for (let step = 0; step < MAX_STEPS; step++) {
    const response = await openai.responses.create({
      model: MODEL,
      instructions: "You are a helpful assistant with access to tools.",
      input,
      tools,
    });

    const calls = response.output.filter(
      (item): item is OpenAI.Responses.ResponseFunctionToolCall => item.type === "function_call",
    );

    // If no tool calls, return the response
    if (calls.length === 0) {
      return response.output_text;
    }

    // Send back everything the model produced (reasoning items included), then each result
    input.push(...(response.output as OpenAI.Responses.ResponseInputItem[]));
    for (const call of calls) {
      input.push({
        type: "function_call_output",
        call_id: call.call_id,
        output: runTool(call.name, JSON.parse(call.arguments)),
      });
    }
  }
  return `Stopped after ${MAX_STEPS} steps without a final answer.`;
}

// Usage
async function main() {
  console.log(await agentLoop("What's the weather in Paris?"));
  console.log(await agentLoop("Search for deployment docs"));
}

main();
