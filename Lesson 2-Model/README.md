# MinIO demo: store Hugging Face dataset as JSONL

This folder shows how to:

- **Run MinIO locally** with Docker Compose.
- **Load a dataset** from Hugging Face (`Salesforce/xlam-function-calling-60k`).
- **Save a JSONL sample** into MinIO.
- **View objects** in the MinIO web console.
- **Set up Python** via conda and Docker Desktop (Windows/macOS) or Docker CLI (Linux).

---

## 1. Requirements

- **Docker**:
  - **Windows / macOS**: Install **Docker Desktop** from the Docker website.
  - **Linux**: Install Docker Engine via your distro package manager (e.g. `apt`, `dnf`, `pacman`).
- **Python 3.9+**.
- (Recommended) **conda** (Miniconda or Anaconda) to manage a virtual environment.

---

## 2. Start MinIO with Docker Compose

From the `minio` folder:

```bash
cd path/to/minio
docker compose up -d
```

What this does:

- Starts a MinIO server at `http://localhost:9000` (S3 API).
- Starts the MinIO web console at `http://localhost:9001`.
- Uses credentials:
  - **User**: `minioadmin`
  - **Password**: `minioadmin123`

To stop MinIO:

```bash
docker compose down
```

---

## 3. Set up Python environment (conda)

### 3.1 Create and activate env (Windows, macOS, Linux)

From `minio/`:

```bash
conda create -n minio-demo python=3.10 -y
conda activate minio-demo
pip install -r requirements.txt
```

`requirements.txt` installs:

- `datasets` – Hugging Face Datasets.
- `minio` – Python client for MinIO / S3.

---

## 4. Upload JSONL sample from Hugging Face dataset

The demo script:

- Loads `Salesforce/xlam-function-calling-60k` from Hugging Face.
- Takes the first 100 rows (`train[:100]`).
- Converts them to **JSONL**.
- Uploads the file to MinIO.

Run (from the `minio` folder):

```bash
python demo_save_jsonl_to_minio.py
```

Defaults (can be overridden with env vars):

- **Endpoint**: `localhost:9000`
- **Access key**: `minioadmin`
- **Secret key**: `minioadmin123`
- **Bucket**: `xlam-data` (created automatically if needed)
- **Object name**: `xlam-function-calling-60k-sample.jsonl`

Environment variables you can set:

```bash
export MINIO_ENDPOINT="localhost:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin123"
export MINIO_BUCKET="xlam-data"
export MINIO_OBJECT_NAME="xlam-function-calling-60k-sample.jsonl"
python demo_save_jsonl_to_minio.py
```

On Windows PowerShell:

```powershell
$env:MINIO_ENDPOINT="localhost:9000"
$env:MINIO_ACCESS_KEY="minioadmin"
$env:MINIO_SECRET_KEY="minioadmin123"
$env:MINIO_BUCKET="xlam-data"
$env:MINIO_OBJECT_NAME="xlam-function-calling-60k-sample.jsonl"
python demo_save_jsonl_to_minio.py
```

---

## 5. View data in the MinIO web console

1. Open the console in your browser:

   - `http://localhost:9001`

2. Log in:

   - **User**: `minioadmin`
   - **Password**: `minioadmin123`

3. In the left sidebar, click **Buckets**.
4. Open the bucket **`xlam-data`**.
5. Click the object `xlam-function-calling-60k-sample.jsonl`.
6. Use:
   - **Overview / Metadata** to see size, content type, etc.
   - **Download** to get the JSONL file locally (the console may or may not support inline preview for `.jsonl`; downloading is always available).

> Note: The MinIO console preview support depends on file extension and content type. If you see “Preview unavailable”, use **Download**, `mc cat`, or a presigned URL instead.

---

## 6. Quick platform notes

### 6.1 Windows (Docker Desktop + conda)

- Install **Docker Desktop**.
- Install **Miniconda** or **Anaconda**.
- Use **Anaconda Prompt** or **PowerShell**:

  ```powershell
  cd path\to\minio
  docker compose up -d

  conda create -n minio-demo python=3.10 -y
  conda activate minio-demo
  pip install -r requirements.txt
  python demo_save_jsonl_to_minio.py
  ```

### 6.2 macOS (Docker Desktop + conda)

```bash
cd path/to/minio
docker compose up -d

conda create -n minio-demo python=3.10 -y
conda activate minio-demo
pip install -r requirements.txt
python demo_save_jsonl_to_minio.py
```

### 6.3 Linux (Docker CLI + conda)

```bash
cd path/to/minio
docker compose up -d

conda create -n minio-demo python=3.10 -y
conda activate minio-demo
pip install -r requirements.txt
python demo_save_jsonl_to_minio.py
```

---

## 7. Optional: CLI preview with `mc`

If you install the MinIO client (`mc`), you can stream a few lines without downloading a file:

```bash
mc alias set localminio http://localhost:9000 minioadmin minioadmin123
mc cat localminio/xlam-data/xlam-function-calling-60k-sample.jsonl | head -n 5
```

This is useful when the console preview is unavailable for a given file type.

