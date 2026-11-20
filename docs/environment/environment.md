# Environment details

- Python: 3.13.16 (CPython), isolated uv virtual environment
- OS: Windows 11 (AMD64)
- Dependency pins: `requirements-lock.txt` (exact); `requirements.txt` (direct dependencies)
- LaTeX: MiKTeX-pdfTeX 4.23 (MiKTeX 25.12); Latexmk, John Collins, 9 March 2026. Version 4.88; class IEEEtran (conference)
- Claude models: Claude Code CLI 2.1.309, headless mode, all tools disabled (see generation logs)
- Open-weight runtime: llama.cpp release b11829 (win-vulkan-x64), device Vulkan1 NVIDIA RTX 4050 Laptop; source https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF (qwen2.5-coder-7b-instruct-q4_k_m.gguf), license Apache-2.0
- Open-weight model file SHA-256: `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`
  (weights not redistributed; obtain from the source named above and verify the hash)
- Network: required only for re-acquiring the official TLC files (and, optionally, the legacy 300-row excerpt).
- Windows note: Smart App Control blocked pillow 13 and the compiled fonttools wheel on the development machine; the lock pins pillow 11.3.0 and fonttools is built from source.
