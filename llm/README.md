# The provider layer

_Last verified: 2026-09-26_

One call, many providers. Your Flagship Agent calls `complete()`; this layer
decides what each model will actually accept and sends that.

```python
from llm import CallOptions, Message, complete
from llm.transport import HttpTransport

answer = complete(
    model="claude-sonnet-5",
    messages=[Message.user("What is Acme's refund window?")],
    options=CallOptions(effort="high", temperature=0.7),
    transport=HttpTransport(provider="anthropic"),
)
print(answer.text)
for note in answer.adjustments:
    print(note)
```

```
dropped temperature=0.7: claude-sonnet-5: a non-default temperature, top_p or
top_k returns a 400 on Claude Opus 4.7 and later models, this one included.
```

That last line is the point. Switching models used to mean a 400 error; here it
means a printed reason.

## What the layer knows

Everything comes from **`models.json`**, the model registry — one data file of
every model the course supports, with its prices, context window, whether it
acts on sampling parameters, which effort levels it has, and which API surface
it speaks. Each entry names the vendor page it was read from and the date it was
read. A model we could not confirm from a vendor source is left out.

| Model | Provider | Surface | In $/MTok | Out $/MTok | Context | temperature | effort levels |
|---|---|---|---:|---:|---:|---|---|
| `claude-opus-5-5` | anthropic | Messages | 4.00 | 20.00 | 1,000,000 | rejected | low…max |
| `claude-sonnet-5` | anthropic | Messages | 2.00 | 10.00 | 1,000,000 | rejected | low…max |
| `claude-haiku-4-5` | anthropic | Messages | 1.00 | 5.00 | 200,000 | accepted | none |
| `gpt-6-sol` | openai | Responses | 2.00 | 10.00 | 1,050,000 | rejected | none…max |
| `gpt-6-luna` | openai | Responses | 0.10 | 0.50 | 1,050,000 | rejected | none…max |
| `gemini-3.8-flash` | google | generateContent | 0.75 | 3.75 | 1,048,576 | accepted | low/medium/high |
| `deepseek-flash` | deepseek | Chat Completions | 0.30 | 1.20 | 1,000,000 | ignored | low/high/max |
| `grok-4.7` | xai | Chat Completions | 2.00 | 6.00 | 500,000 | accepted | low…xhigh |

Read `models.json` for the authoritative version, including each entry's source.

## The three things that differ between models

**Sampling parameters.** Claude models from Opus 4.7 on return a 400 for a
non-default `temperature`. Current OpenAI models reject it whenever reasoning
effort is anything but `none`. DeepSeek in thinking mode accepts it and then
ignores it — which the registry treats the same way, because sending a value
that does nothing is a lie about the request. The layer never sends one to any
of them.

**Effort.** Every provider spells "think harder" differently:
`output_config.effort`, `reasoning.effort`, `reasoning_effort`,
`generationConfig.thinkingConfig.thinkingLevel`. You pass one level from
`EFFORT_LADDER`; the layer puts it in the right place. Models with no effort
control drop it. Models missing your exact level snap to the nearest one they
have, ties going to the lower level so a translation never quietly spends more
than you asked for.

**Tool calls.** Current OpenAI models need the **Responses API** for function
calling — Chat Completions only does it at `reasoning_effort: "none"` — so the
registry pins them to that surface. A `ToolSpec` becomes an Anthropic
`input_schema`, an OpenAI `parameters`, or a Gemini `functionDeclarations`
entry; calls come back as normalized `ToolCall`s whichever provider answered.

## One line, for scripts and notebooks

The Appendix calls `ask()`: one user turn in, the answer text out, still through
`complete()`, so the same drops and translations apply.

```python
from llm import ask, default_model

print(default_model())          # which model will answer
print(ask("What is 2+2?", system="Answer in one word.", temperature=0.2))
```

The model is `LLM_MODEL` when set (a registry ID), else the cheapest registry
model of `LLM_PROVIDER`, else the cheapest model of the first provider whose key
is in the environment. Load your `.env` first; the layer reads `os.environ` only.
Use `complete()` when you need tool calls, usage, cost or the adjustments.

## Offline: record once, replay free

```python
from llm import RecordingTransport, ReplayTransport
from llm.transport import HttpTransport

recorder = RecordingTransport(HttpTransport(provider="anthropic"), "refund.recording.json")
...                                     # run the Case once, on your own key
recorder.save()

offline = ReplayTransport("refund.recording.json")   # no key, no network, for ever
```

A recording holds the full prompt, every tool argument and every tool result, so
`*.recording.json` is gitignored everywhere. The only recordings committed are the
reviewed ones the Offline Checks replay, in `company/recordings/`. Read a recording
before you move it there.

A replay answers only a request it has actually seen. Change the prompt, the
tools, or an option that reaches the wire, and it raises `ReplayMismatchError`
naming the fields that moved — a stale recording fails loudly instead of
reporting green for behaviour nobody ran.

## Adding a model (every Drop)

1. Open the vendor's own docs. Not a blog post, not this file.
2. Add an entry to `models.json` with `source` and today's `verified_on`.
3. Run `pytest tests/test_provider_registry.py tests/test_provider_sampling.py
   tests/test_provider_effort.py tests/test_provider_tools.py`. Every registry
   model is checked on its outgoing request, so a new entry is exercised the
   moment it exists.
4. If you cannot confirm a fact from the vendor, leave the model out.

## Notes

- No API keys and no network are needed to run any of the tests: the request
  is built and asserted on before a transport is involved.
- The layer speaks raw JSON over a `Transport` rather than using vendor SDKs,
  because the outgoing request is the seam the Checks assert on and the thing
  the replay transport records. An SDK would hide it.
- `shared/provider.py` is the course's old provider helper. The Appendix code
  and notebooks no longer use it (ticket 14); `demo.py` and the MCP example still
  do until they move, and then it is deleted.
