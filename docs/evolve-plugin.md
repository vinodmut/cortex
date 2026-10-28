# evolve (Self-Improvement Loop) Plugin

The `evolve` plugin connects Cortex to [altk-evolve](https://github.com/AgentToolkit/altk-evolve),
a service that learns *guidelines* — short instructions that would have steered an agent
better — from recorded agent sessions. It closes the loop rossoctl#2653 describes:

1. **Capture** — Cortex already records every session (`core/session/archive`, session API).
2. **Learn** — when a session goes quiet, the plugin tells evolve, which reads the session's
   new turns from Cortex's session API and distils **candidate** guidelines from them.
3. **Inject** — on each main-agent request, the plugin appends the **active** guidelines
   relevant to the session's task to the system instructions.

Evolve is Python, so it runs beside Cortex as a local service (`evolve cortex serve`) and the
plugin is a thin HTTP client — the same shape as [`sparc`](sparc-plugin.md).

## Nothing learned is injected until a person accepts it

Session content includes tool output: fetched web pages, file contents, command output. A
guideline distilled from it can carry text an attacker wrote, and injecting it makes that text
persistent across every later session. So evolve stores everything it learns from Cortex as
`status: candidate`, and serves the plugin only guidelines a reviewer promoted:

```bash
evolve entities list cortex --status candidate     # review what was learned
evolve entities promote cortex <id> --reason "..." # accept: injected from now on
evolve entities retire cortex <id> --reason "..."  # withdraw; content and history are kept
```

A retired guideline stops being injected once each session's cache expires
(`cache_ttl_seconds`, default 30 minutes).

## What a request looks like

The block is appended after the existing system instructions — a trailing `system` text block
on Anthropic Messages, the leading system message on Chat Completions, `instructions` on the
Responses API:

```text
<evolve-guidelines source="altk-evolve" ids="0b5c…,7d21…">
Guidelines learned from earlier sessions on this machine. Apply them when they are relevant to the task; ignore them otherwise.
1. Run the test suite with `uv run pytest -v` before committing.
2. …
</evolve-guidelines>
```

Guidelines are fetched **once per session** (from the session's first user message, with
`<system-reminder>` context removed) and the rendered block is cached, so every turn appends
identical bytes and the provider's prompt cache keeps hitting. Requests the plugin leaves
alone: subagents, tool-less side calls (titles, summaries), sessions without an id, agents
outside `agents`, and its own calls (`X-Evolve-Cortex`).

Inference parsers read the client's original bytes, so the session archive — and therefore
what evolve learns from — never contains the injected block.

## Seeing what it did

Every handled request records an invocation (`modify` when the block was added, `skip` with a
reason otherwise) and publishes `plugins.evolve` on its session event:

```json
{"injected": true, "ids": ["0b5c…", "7d21…"], "revisions": [1, 3],
 "sources": ["cortex:<session>:<seq>", null], "cached": true, "fetchedAt": "…"}
```

`ids` and `revisions` name exactly which guideline versions shaped the request; `sources` is
the session turn each one was learned from. agentop's detail pane shows the event; the plugin
pane shows counters (`requests injected`, `fetch errors`, `idle sessions reported`, …).

## Failure policy: open

Evolve unreachable, slow (`timeout_ms`, default 1.5 s, paid once per session), or returning an
error means the request is forwarded **unchanged**, a fetch error is counted, and the plugin
tries again after `retry_after_seconds`. A session that already has guidelines keeps them
through a failed refresh. Idle-session reports that fail are retried on the next 30-second tick;
evolve's per-session cursor makes a repeated report harmless. Sessions never reported (Cortex
restarted, evolve down) are picked up by `evolve sync cortex`.

## Configuration

```yaml
pipeline:
  outbound:
    plugins:
      - name: inference-parser
      - name: mcp-parser
      - name: a2a-parser
      - name: tool-prune
      - name: evolve
        config:
          endpoint: http://127.0.0.1:8202
          bearer_file: ~/.evolve/cortex-token
          agents: [claude-code]
```

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | required | Base URL of `evolve cortex serve`. |
| `bearer` / `bearer_file` | — | The token evolve was started with. Use one; prefer the file or `${ENV}`. |
| `inject` | `true` | Append active guidelines to main-agent requests. |
| `export` | `true` | Report idle sessions so evolve learns from them. |
| `agents` | all | Client names to act on (`claude-code`, `codex`, `bob`, `opencode`). |
| `top_k` | `8` | Most guidelines injected per session. |
| `timeout_ms` | `1500` | Per-call timeout. |
| `cache_ttl_seconds` | `1800` | How long a session's guidelines are reused. |
| `retry_after_seconds` | `60` | Back-off after a failed fetch. |
| `idle_after_seconds` | `900` | Quiet time before a session is reported. |
| `max_sessions` | `1024` | Sessions whose guidelines are cached. |

`on_error: observe` computes the block and publishes the event without changing any bytes.

## Running evolve beside Cortex

```bash
umask 077; mkdir -p ~/.evolve; openssl rand -hex 32 > ~/.evolve/cortex-token
env -u HTTPS_PROXY \
  EVOLVE_API_TOKEN_FILE=~/.evolve/cortex-token \
  EVOLVE_NAMESPACE_ID=cortex EVOLVE_BACKEND=filesystem EVOLVE_DATA_DIR=~/.evolve/data \
  EVOLVE_CORTEX_URL=http://127.0.0.1:47601 \
  EVOLVE_MODEL_NAME=<litellm model for guideline generation> \
  evolve cortex serve --port 8202
```

`evolve cortex serve` refuses to start without a token and serves only the Cortex API — not
evolve's unauthenticated dashboard. Unset `HTTPS_PROXY` so evolve's own LLM calls do not
route through Cortex and get recorded as agent traffic.

## Turning it off

Remove the `evolve` entry (or set `on_error: off`); the config hot-reloads. Learned guidelines
stay in evolve; retiring them there is the per-guideline off switch.

## Not yet

- **Shadow evaluation.** Adoption is a manual promotion today. Injecting a candidate on a
  fraction of sessions and promoting it only when cost and an outcome signal strictly beat the
  baseline (rossoctl#2653, SkillOpt's gate) is the next step.
- Redaction of session content before it reaches evolve's LLM (evolve's hooks can do this but
  are opt-in), and subagent trajectories.
- Ordering with `context-guru`, which may compact the block away; place `evolve` after it.
