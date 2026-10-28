# Changes for the standalone extraction

## 0.1.0 — 28 October 2026

- Replace ATMAN imports with package-relative dependencies.
- Keep project paths, tennis postprocessing and ATMAN data adapters in ATMAN.
- Add explicit-path grammar parsing and a small public API.
- Replace import-time NumPy error-policy changes with local handling and use stable logaddexp for log-probability sums.
- Embedded re-estimation now counts same-frame entry/exit transitions on the final frame. The old slicing dropped the final model’s exit, producing a substochastic row and NaNs in the next cycle. Synthetic two-model training and transition-normalisation tests cover the correction.
- Shared recursive-grammar links reuse an already-recorded identical terminal exit on the same frame. This removes duplicate zero-duration events without changing the token likelihood or predecessor. A repeated two-label grammar test covers the backtrace.
- Transition normalisation handles the silent exit row explicitly instead of raising an invalid-division warning.
