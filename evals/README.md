# evals/

Hand-verified evaluation data (not unit-test fixtures — those live in `tests/`).

- `voice_labels_T0001889_c225.json` — human labels (audio + video verified) for the
  voice-search demo hits, query = T0001889_00001.mp4 chunk 225. Sections: `cross`
  (top-15 cross-video hits; `same` = verified same person as the query) and `same`
  (top-8 same-video hits; `diff` = a *different* speaker in the same recording —
  direct evidence of channel inflation). Exported from
  `audio-search-demo/voice_search_demo_labeled.html` (generator: `gen_labeled.py`).
