---
name: intent-routing
description: >
  Classifies user messages into PARSE, INGEST, RETRIEVE, PIPELINE, or AMBIGUOUS
  and routes to the correct sub-agent with fully assembled parameters. Use when
  receiving any user message that references documents, files, or content — which
  is every non-trivial turn. Provides the routing decision table, parameter
  assembly rules for each sub-agent, and ambiguity resolution patterns. Activate
  on every turn where intent classification is needed.
metadata:
  author: Chanoch Clerk Pipeline
  version: "1.0"
---

# Intent Routing Skill

## When to activate
Every turn where the user references a file, document, or content query.

## Intent Classification

| Intent | Key signals | Delegate to |
|--------|-------------|-------------|
| PARSE | file path, attachment, "parse", "extract", "read this file", "what's in" | document_parser_agent |
| INGEST | "ingest", "store", "index", "add to system", "verify", "integrity", "history", "versions", "delete", "purge" | ingestion_agent |
| RETRIEVE | question about content, "find", "search", "most relevant", "what does X say about" | reranker_agent |
| PIPELINE | "parse and ingest", "process and store", "full pipeline", new file + content question in same message | document_parser_agent → ingestion_agent → reranker_agent |
| AMBIGUOUS | unclear target, "do something with", no specific action, multiple possible intents | ask + offer options |

## Steps — Single Intent

1. Classify intent from the table above.
2. Resolve file reference: use explicit path/filename, or fall back to
   `orchestrator:active_file` from state.
3. Assemble sub-agent delegation parameters (see below).
4. For write ops (PARSE, INGEST): confirm in one sentence before delegating.
   For read ops (RETRIEVE): delegate immediately.
5. After response: translate result to plain English. Update
   `orchestrator:active_file` and `orchestrator:last_intent` in state.

## Steps — PIPELINE (multi-step)

1. Announce all steps upfront: "I'll parse, ingest, then search — starting now."
2. Set `orchestrator:pipeline_step = 1`.
3. Delegate PARSE. On success: announce completion + next step. Increment step.
4. Delegate INGEST with `file_path` from parser output
   (absolute path `src/{stem}/documents.json`). On success: announce + increment.
5. Delegate RETRIEVE with the user's original query. Present results.
6. On any step failure: report which step failed, what succeeded, and next action.
   Do not silently continue.

## Steps — AMBIGUOUS

1. Identify what is known (filename, partial intent).
2. Respond with one question + 2–4 labelled options:
   "What would you like to do with {file}?
   a) Parse it   b) Ingest it   c) Search it   d) Check integrity"
3. Wait for user selection. Do not guess.

## Parameter Assembly

**document_parser_agent:**
- `file_path`: resolved absolute path or base64+filename
- Set `parser:active_category` in state if user mentioned a category

**ingestion_agent:**
- Ingest: `file_path = absolute path to src/{stem}/documents.json` (parser output path)
- Search: `query` (stripped of conversational framing), `category`, `version_root`
- Purge: `filename` + warn user + wait for confirmation → set `ingestor:purge_confirmed=True`
- Audit: `filename` + `page_index` (1-indexed)

**reranker_agent:**
- `query`: extract core query, strip "what does the document say about", "find", etc.
- `rerank_top_n`: default 5; use 10 if user asks for "more results"
- `category`: from `orchestrator:active_category` if set
- `include_citations_text`: True when user needs formatted citations

## Result Formatting

**PARSE result:** "Parsed {page_count} page(s) from {filename}. Would you like to ingest it?"

**INGEST result:** "Stored {ingested} page(s) ({skipped} unchanged). Ready to search."
If errors: "Stored {ingested} page(s), but {n} failed — {first_error_summary}."

**RETRIEVE result:** numbered list —
```
1. {filename} · p.{page_index} · score {final_score:.2f}
   "{content[:120]}…"
```
End with: "Would you like more detail on any of these?"

**AUDIT result:** "Integrity check {PASSED/FAILED} for {filename} page {page_index}."
On FAILED: "The stored data may be out of sync. I'll attempt recovery automatically."
[then delegate ingest_sync to ingestion_agent]

## Gotchas

- Strip conversational framing from queries before passing to reranker_agent.
  "What does the policy say about X?" → query = "X".
- Parser output path convention: absolute path to `src/{document_stem}/documents.json`. 
  This is automatically discovered by callbacks if the file exists.
- A user asking "search it" right after parsing — without ingesting — will get
  empty results. Detect this and offer to ingest first.
- version_root values are 64-char hex strings. Never display them fully to users;
  use "version from {timestamp}" as the user-facing label.
- Read `references/sub-agent-contracts.md` for the exact input/output schemas
  of each sub-agent if assembling complex delegations.

## Constraints

- Never delegate to multiple sub-agents in the same turn — always sequential.
- Never expose sub-agent names, Merkle roots, UUIDs, or raw JSON to users
  unless explicitly requested.
- Never proceed with purge without user confirmation in the current turn.