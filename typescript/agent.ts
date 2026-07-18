import OpenAI from "openai";

const openai = new OpenAI();

// Define tools
const tools = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get current weather for a location",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string", description: "City name" },
        },
        required: ["location"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "search_docs",
      description: "Search internal documentation",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
        },
        required: ["query"],
      },
    },
  },
];

// Tool implementations
function getWeather(location: string): string {
  return `Weather in ${location}: 72°F, sunny`;
}

function searchDocs(query: string): string {
  return `Found 3 docs about "${query}"`;
}

async function agentLoop(userMessage: string): Promise<string> {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: "You are a helpful assistant with access to tools." },
    { role: "user", content: userMessage },
  ];

  while (true) {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      tools,
    });

    const choice = response.choices[0];

    // If no tool calls, return the response
    if (!choice.message.tool_calls) {
      return choice.message.content ?? "";
    }

    // Execute tool calls
    messages.push(choice.message);
    for (const toolCall of choice.message.tool_calls) {
      const args = JSON.parse(toolCall.function.arguments);
      let result: string;

      if (toolCall.function.name === "get_weather") {
        result = getWeather(args.location);
      } else if (toolCall.function.name === "search_docs") {
        result = searchDocs(args.query);
      } else {
        result = "Unknown tool";
      }

      messages.push({
        role: "tool",
        tool_call_id: toolCall.id,
        content: result,
      });
    }
  }
}

// Usage
async function main() {
  console.log(await agentLoop("What's the weather in Paris?"));
  console.log(await agentLoop("Search for deployment docs"));
}

main();
