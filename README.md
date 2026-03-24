# LLMOps

A simple repository for LLMOps practice projects: data pipelines, local object storage demos, and training tutorial code.

## Repository Structure

- `Two-Model/` - MinIO + Python demo to save dataset samples as JSONL.
- `Lesson 2_Model/` - lesson-based version of the MinIO workflow.
- `llm_training_tutorial/` - model training tutorial code and scripts.

## Requirements

- Python 3.9+ (Python 3.10 recommended)
- Docker Desktop (Windows/macOS) or Docker Engine (Linux)


## Notes

- If a Hugging Face dataset is gated, access may be required.  
  The current demo script can fall back to a public dataset so the upload flow still works.
- Do not commit API keys or tokens. Use environment variables and `.gitignore`.

## Stop Services

```powershell
cd "C:\Users\anhhu\Downloads\LLMOps\Two-Model"
docker compose down
```
