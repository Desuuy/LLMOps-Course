# LLMOps

Repo này chứa các bài thực hành LLMOps (Docker, dữ liệu, model demo).

## Cấu trúc chính

- `Two-Model/`: demo lưu dữ liệu JSONL lên MinIO bằng Python.
- `Lesson 2_Model/`: bản bài học tương tự theo từng bước.
- `llm_training_tutorial/`: mã nguồn liên quan tới train model.

## Yêu cầu

- Python 3.9+ (khuyên dùng 3.10)
- Docker Desktop (Windows/macOS) hoặc Docker Engine (Linux)

## Chạy nhanh demo MinIO (Two-Model)

Mở PowerShell:

```powershell
cd "C:\Users\anhhu\Downloads\LLMOps\Two-Model"
pip install -r requirements.txt
docker compose up -d
python demo_save_jsonl_to_minio.py
```

Sau khi chạy xong:

- MinIO Console: `http://localhost:9001`
- User: `minioadmin`
- Password: `minioadmin123`
- Bucket mặc định: `xlam-data`

## Ghi chú

- Nếu dataset Hugging Face bị gated, script sẽ fallback sang dataset public để demo vẫn chạy.
- Không commit token/API key vào git. Dùng `.gitignore` và biến môi trường.

## Dừng dịch vụ

```powershell
cd "C:\Users\anhhu\Downloads\LLMOps\Two-Model"
docker compose down
```
