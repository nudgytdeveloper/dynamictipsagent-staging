# Changelog

## 2026-10-28

### Fixed
- When Gemini has not answered within `TIPS_HEDGE_SECONDS` (5 s) or fails sooner, OpenAI is asked too and the first tip back is returned; Gemini and OpenAI timeouts default to 20 s (were 60 s)

### Changed
- A PDF SOP with at least `SOP_PDF_MIN_TEXT_CHARS` (500) characters of text is sent to the model as text, not as the PDF; `pypdf` added

## 2026-10-26

### Fixed
- `anchor_quote` is filled for short learner messages such as "hello", which were below the 8-character minimum meant only for searching every message; quotes now match whole words (task 295)
- When the model corrects the learner's typos while quoting, the closest run of the learner's own words in the named message is stored as `anchor_quote`, in their spelling (task 295)

### Changed
- The prompt asks for the anchor quote character for character, keeping spelling mistakes

## 2026-10-21

### Added
- `/tips` accepts an optional `is_requested` (body, or query parameter on `/tips/{simulation_id}`), stored on each tip entry and echoed in the response; defaults to `false`, which shows the notification bell

### Fixed
- New tip documents store `simulation_id` as an ObjectId, which the platform reads; tips for a simulation are appended to its existing document whether it is keyed by an ObjectId or the older string

## 2026-10-13

### Added
- Each stored tip entry gets an `_id` so the platform can edit, delete or comment on a single tip (task 295)
- Each stored tip entry records the transcript line it is about: `anchor_transcript_id`, `anchor_quote` (the learner's exact words, for highlighting) and `anchor_method` (`model`, `quote_match` or `latest`), plus `last_transcript_id` at generation time
- `/tips` response gains `tip_id`, `anchor_transcript_id`, `anchor_quote` and `anchor_method`; the timing log line gains `anchor=`

### Changed
- Learner messages are labelled `[U1]`, `[U2]`, … in the prompt and the model names the one its tip is about on a trailing `ANCHOR` line, which is stripped before the tip is stored; stored tip text keeps its existing format

## 2026-10-08

### Added
- OpenAI fallback: when Gemini errors or returns no text, the same prompt and SOP PDF go to OpenAI (`OPENAI_API_KEY`, `OPENAI_FALLBACK_MODEL` default `gpt-4o-mini`, `OPENAI_TIMEOUT_SECONDS` default `60`); without the key nothing changes
- `model` on the `/tips` response and in the timing log line: the model that wrote the tip

## 2026-10-07

### Changed
- `GEMINI_MODEL` defaults to `gemini-3.8-flash`; the replacement Gemini key cannot call `gemini-2.5-flash`
- A negative `GEMINI_THINKING_BUDGET` leaves `thinking_config` off the request

## 2026-09-01

### Changed
- Gemini is called with thinking switched off (`GEMINI_THINKING_BUDGET`, default `0`); a single ≤500-character tip does not need the model to reason first, and that reasoning was most of the 9.4 s median / 12.4 s p90 measured on staging 2026-08-25 → 08-31
- The agent now uses the `google-genai` SDK; `google-generativeai` is end-of-life and had no way to set a thinking budget
- The SOP file is downloaded once per URL and cached (`SOP_CACHE_TTL_SECONDS`, default 600; an unusable file is remembered for `SOP_CACHE_FAILURE_TTL_SECONDS`, default 60) instead of on every request
- Transcript and simulation lookups run concurrently, as do the Continuous Learning lookup and the SOP download
- The three Continuous Learning audit writes run after the response is sent

### Added
- `duration_ms` on the `/tips` response, and one `Tips for simulation …: <total>ms (lookup= context= gemini= save=)` log line per request
- `GEMINI_MODEL` (default `gemini-2.5-flash`) and `GEMINI_TIMEOUT_SECONDS` (default `60`) env vars

### Fixed
- A Gemini response with no text (safety block, or the whole budget spent on thinking) is returned as an error payload instead of raising
- Log lines reach Render as they happen instead of in one clump at the end of the request

## 2026-08-24

### Added
- Tips follow the learner's Continuous Learning instruction: the agent reads the active `agent_learner_instructions` document for the simulation's learner and injects it into the prompt after the SOP block, so a retry targets the learner's weakest skills first and a first attempt gets simple, encouraging tips
- Each persisted tip entry records `cl_instruction_id`, `cl_instruction_version` and `cl_stage`; the `/tips` response gains `cl_applied`
- `CL_INSTRUCTIONS_COLLECTION` env var (default `agent_learner_instructions`)

## 2026-08-11

### Added
- Tips are now grounded in the service's uploaded SOP: the agent resolves the simulation's `sop_tips_generation_file` from `dependency_snapshot` (falling back to the live `service_levels` doc), downloads the file, and passes a PDF to Gemini as inline data (or inlines a `.txt`/`.md` body), instructing the model to prefer a tip about an SOP instruction the learner did not follow over generic communication advice — task 307
- `SIMULATIONS_COLLECTION` and `SERVICE_LEVELS_COLLECTION` env vars (defaulting to `simulations` / `service_levels`) so the agent can resolve a simulation to its service's SOP
- `sop_applied` and `sop_file_url` on the `/tips` response and on each persisted tip entry, so a tip can be traced back to the document that produced it

### Fixed
- SOP file URLs are percent-encoded before download. Uploaded SOP filenames routinely contain spaces (e.g. `Tips SOP Sample .pdf`), which `urllib` rejects with `http.client.InvalidURL`; found by the first end-to-end run against a real staging simulation, which returned HTTP 500
- `fetch_sop_document` caught only `URLError`/`HTTPError`/`OSError`/`ValueError`, but `http.client.InvalidURL` descends from `Exception` alone, so a bad SOP URL escaped as an unhandled 500 instead of degrading. It now catches `Exception`, honouring its documented "never raises" contract
- `generate_tips` returned `[]` on a generation failure while `TipsResponse.tips` is typed `str`, turning every Gemini error into a pydantic validation error (HTTP 500) instead of a clean error payload; it now returns `""`

### Changed
- A missing simulation, missing SOP field, unreachable URL, oversized file (>15 MB), or unsupported file type logs and falls back to the previous generic tip instead of failing the request
