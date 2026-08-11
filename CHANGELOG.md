# Changelog

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
