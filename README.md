# Q-Stream: Query-Aware Streaming Video Understanding

Q-Stream is a hybrid local + GPU web application for streaming video understanding. The browser interface, authentication, SQLite database, bookmarks, and camera-recording management run on the local computer, while the AI pipeline runs on a rented Linux GPU server.

The remote AI backend performs frame sampling, OpenCLIP encoding, event construction, Qwen2.5-VL event description, 7WH query-profile prediction, memory-action prediction, evidence retrieval, and timestamp-grounded question answering.

---

## 1. Recommended Setup

### Local computer

- Windows 10/11
- Python 3.10 or later
- Google Chrome or Microsoft Edge
- Stable internet connection
- Webcam if testing live-camera mode

### Remote GPU server

A CUDA-capable NVIDIA GPU is required for practical AI inference.

**Recommended:** rent an NVIDIA RTX 5090 instance from AutoDL. A similar GPU cloud provider can also be used. The SSH host and port are different for every rented instance, so replace the example values in this README with the values shown by your provider.

The examples below use:

```text
SSH host: connect.bjb1.seetacloud.com
SSH port: [your ssh port]
Remote AI port: 8000
Local web port: 5500
Ollama port: 11434
```

> The `seetacloud.com` address above is only an example from one GPU instance. If AutoDL is used, use the SSH address and port provided by AutoDL instead.

---

## 2. Project Structure

Important folders/files:

```text
A3v1/
├── main.html
├── login.html
├── signup.html
├── savedLog.html
├── server.py
├── database/
├── recordings/
└── Q_Stream_backend/
    └── Q_Stream/
        ├── qstream_api_server.py
        ├── setup_env.sh
        ├── requirements.txt
        ├── pipeline_stage1.py
        ├── event_construction.py
        ├── event_description_ollama.py
        ├── data/coin/COIN.json
        └── training/models/
```

### Important remote path

The current backend code uses this path:

```text
/root/autodl-tmp/Q_Stream
```

Therefore, the folder:

```text
A3v1/Q_Stream_backend/Q_Stream
```

should be uploaded to the GPU server as:

```text
/root/autodl-tmp/Q_Stream
```

If the backend is stored somewhere else, update `PROJECT_DIR` in `qstream_api_server.py` before running it.

---

# 3. First-Time GPU Setup

## Step 1 - Rent and start the GPU instance

Rent a Linux GPU instance. An RTX 5090 is preferred for faster Qwen2.5-VL and OpenCLIP inference.

Choose an image/environment with Python, CUDA, and PyTorch support if possible. Q-Stream should not be run as CPU-only for normal use.

After the instance starts, note its:

- SSH hostname
- SSH port
- username, normally `root`

---

## Step 2 - Upload the Q-Stream backend

Upload this local folder:

```text
A3v1/Q_Stream_backend/Q_Stream
```

to:

```text
/root/autodl-tmp/Q_Stream
```

You may use the cloud provider file manager, SFTP, or SCP.

Example SCP command from Windows PowerShell:

```powershell
scp -P [your ssh port] -r ".\Q_Stream_backend\Q_Stream" root@connect.bjb1.seetacloud.com:/root/autodl-tmp/
```

After uploading, the remote server should contain:

```text
/root/autodl-tmp/Q_Stream/qstream_api_server.py
/root/autodl-tmp/Q_Stream/setup_env.sh
/root/autodl-tmp/Q_Stream/training/models/qstream_model.joblib
```

---

## Step 3 - Connect to the GPU server

Open PowerShell or Windows Terminal:

```powershell
ssh -p [your ssh port] root@connect.bjb1.seetacloud.com
```

Replace the hostname and port with your own GPU-instance connection information.

Then enter the backend directory:

```bash
cd /root/autodl-tmp/Q_Stream
```

---

## Step 4 - Install Q-Stream dependencies

Run:

```bash
chmod +x setup_env.sh
bash setup_env.sh
```

The script installs the main Linux system packages and Python packages required by the AI backend, including OpenCV, OpenCLIP, FastAPI, Uvicorn, scikit-learn, joblib, requests, imbalanced-learn, and `python-multipart`.

### Dependency check

The current `qstream_api_server.py` imports `pandas`, but the provided `setup_env.sh`/`requirements.txt` does not explicitly install it. Install it once after running the setup script:

```bash
pip install pandas
```

It is recommended to also verify the main runtime imports:

```bash
python -c "import torch, cv2, open_clip, numpy, pandas, sklearn, joblib, requests, fastapi, uvicorn; print('Q-Stream Python dependencies OK'); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected result:

```text
Q-Stream Python dependencies OK
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
```

The exact GPU name depends on the rented instance.

> If `CUDA available` is `False`, use a CUDA/PyTorch GPU image or repair the PyTorch CUDA installation before continuing.

---

# 4. Install and Start Qwen2.5-VL with Ollama

Q-Stream uses:

```text
qwen2.5vl:7b
```

through Ollama.

The provided `setup_env.sh` checks whether Ollama exists, but it does **not** guarantee that Ollama itself is installed or that the Qwen model has already been downloaded.

Check Ollama:

```bash
ollama --version
```

If Ollama is already installed, start its service:

```bash
ollama serve > /tmp/ollama.log 2>&1 &
```

Check the server:

```bash
curl http://localhost:11434/api/tags
```

Then download the model:

```bash
ollama pull qwen2.5vl:7b
```

Verify it:

```bash
ollama list
```

`qwen2.5vl:7b` should appear in the model list.

If `ollama` is not installed, install Ollama on the GPU server first, then repeat the commands above.

---

# 5. COIN Dataset Videos

The repository already contains:

```text
data/coin/COIN.json
```

but the actual COIN `.mp4` files may need to be copied separately.

For the built-in COIN video dropdown, place videos in:

```text
/root/autodl-tmp/Q_Stream/data/coin/videos/
```

Example:

```text
/root/autodl-tmp/Q_Stream/data/coin/videos/0R9pdc9dO3Q.mp4
```

Check the folder:

```bash
ls -lh /root/autodl-tmp/Q_Stream/data/coin/videos/
```

If no COIN videos are present, the frontend's COIN video list will be empty. Live-camera mode can still be used when the backend is running.

---

# 6. Start the Remote Q-Stream AI Backend

On the GPU server:

```bash
cd /root/autodl-tmp/Q_Stream
python qstream_api_server.py
```

The backend should start on:

```text
http://localhost:8000
```

You should see output similar to:

```text
Q-Stream API build: qstream-api-v1
Streaming mode: background-worker
...
Uvicorn running on http://0.0.0.0:8000
```

### Verify the backend on the remote server

In another remote shell, or before starting it in the foreground, you can check:

```bash
curl http://localhost:8000/health
```

and:

```bash
curl http://localhost:8000/model_status
```

`/model_status` should report that the model bundle exists and can be loaded from:

```text
/root/autodl-tmp/Q_Stream/training/models/qstream_model.joblib
```

Keep the backend running.

---

# 7. Create the SSH Tunnel to Port 8000

The browser is configured to call:

```text
http://localhost:8000
```

Therefore, port `8000` on the rented GPU server must be forwarded to port `8000` on the local Windows machine.

Open a **second PowerShell/Terminal window** on Windows and run:

```powershell
ssh -N -p [your ssh port] -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com
```

Keep this terminal open while using Q-Stream.

You may also omit `-N` if you want an interactive remote shell:

```powershell
ssh -p [your ssh port] -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com
```

### Verify the tunnel from Windows

Open:

```text
http://localhost:8000/health
```

in your browser, or run:

```powershell
curl http://localhost:8000/health
```

If a JSON response is returned, the tunnel is working.

---

# 8. Start the Local Web Application

Open a **third PowerShell/Terminal window** on Windows.

Go to the local project folder:

```powershell
cd "D:\UOW\Degree\degree y3s2\IP1\A3\A3v1"
```

The local server imports the `requests` package. If it has not been installed on the Windows environment, run:

```powershell
pip install requests
```

If you plan to use the local YouTube-download endpoint, also install:

```powershell
pip install yt-dlp
```

and make sure FFmpeg is installed and available in `PATH` because `yt-dlp` may require it to merge video and audio streams.

Start the local application server:

```powershell
python server.py --port 5500
```

Expected output:

```text
Serving AI Video Query at http://localhost:5500
SQLite database: ...\database\app.sqlite3
Camera recordings: ...\recordings
```

Keep this terminal open.

---

# 9. Open Q-Stream

Open the following URL in Chrome or Edge:

```text
http://localhost:5500/main.html
```

If you are not logged in, the application will redirect you to the login page.

For a new installation, create an account using the Sign Up page and then enter the main Q-Stream interface.

---

# 10. Normal Startup After the First Installation

After dependencies and Qwen2.5-VL have already been installed, you normally need **three terminals**.

## Terminal 1 - Remote GPU backend

```powershell
ssh -p [your ssh port] root@connect.bjb1.seetacloud.com
```

Then on the remote server:

```bash
cd /root/autodl-tmp/Q_Stream
ollama serve > /tmp/ollama.log 2>&1 &
python qstream_api_server.py
```

## Terminal 2 - SSH port forwarding

```powershell
ssh -N -p [your ssh port] -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com
```

## Terminal 3 - Local Q-Stream server

```powershell
cd "D:\UOW\Degree\degree y3s2\IP1\A3\A3v1"
python server.py --port 5500
```

Then open:

```text
http://localhost:5500/main.html
```

---

# 11. How the Connection Works

```text
Browser
  │
  ├── http://localhost:5500
  │      Local server.py
  │      ├── login / signup
  │      ├── SQLite database
  │      ├── bookmarks
  │      └── camera recording storage
  │
  └── http://localhost:8000
         SSH tunnel
              │
              ▼
         Remote GPU :8000
         qstream_api_server.py
              │
              ├── OpenCLIP
              ├── Q-Stream trained models
              └── Ollama :11434
                    └── qwen2.5vl:7b
```

The frontend currently contains:

```javascript
const APP_BASE = "http://localhost:5500";
const AI_BASE = "http://localhost:8000";
```

This is why both local ports must be available.

---

# 12. Common Problems

## `ModuleNotFoundError: No module named 'pandas'`

Install the missing runtime dependency:

```bash
pip install pandas
```

Then restart:

```bash
python qstream_api_server.py
```

---

## `Form data requires "python-multipart" to be installed`

Run:

```bash
pip install python-multipart
```

The provided `setup_env.sh` already attempts to install this package.

---

## `ollama: command not found`

Ollama is not installed on the remote GPU server. Install Ollama first, then run:

```bash
ollama serve > /tmp/ollama.log 2>&1 &
ollama pull qwen2.5vl:7b
```

---

## Qwen model not found

Check:

```bash
ollama list
```

If `qwen2.5vl:7b` is missing:

```bash
ollama pull qwen2.5vl:7b
```

---

## Browser cannot connect to `localhost:8000`

Check all of the following:

1. The remote backend is running:

   ```bash
   curl http://localhost:8000/health
   ```

2. The SSH tunnel is still open:

   ```powershell
   ssh -N -p [your ssh port] -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com
   ```

3. Test from Windows:

   ```powershell
   curl http://localhost:8000/health
   ```

---

## Port 8000 is already in use on Windows

Check the process:

```powershell
netstat -ano | findstr :8000
```

Stop the existing process or change the local forwarded port. If the port is changed, `AI_BASE` in `main.html` must be updated to match.

---

## Port 5500 is already in use

Run the local server on another port:

```powershell
python server.py --port 5501
```

However, the current `main.html` contains:

```javascript
const APP_BASE = "http://localhost:5500";
```

so if you change the local application port, update `APP_BASE` in `main.html` as well.

---

## COIN video dropdown is empty

Verify that `.mp4` files exist on the GPU server:

```bash
ls /root/autodl-tmp/Q_Stream/data/coin/videos/
```

The `/videos` API only lists `.mp4` files from this directory.

---

## `CUDA available: False`

Verify the GPU:

```bash
nvidia-smi
```

Then check PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Use a CUDA-enabled PyTorch environment before starting Q-Stream.

---

## Live camera does not start

Use Chrome or Edge and allow camera permission for:

```text
http://localhost:5500
```

The browser sends one camera frame to the remote backend approximately every 3 seconds while camera streaming is active.

---

# 13. Dependency Audit of the Provided Setup

The current project setup is mostly complete, but the following points are important:

| Item | Status | Notes |
|---|---|---|
| `opencv-python` | Included | Backend frame/video processing |
| `pillow` | Included | Image handling |
| `numpy` | Included | Numerical operations |
| `open_clip_torch` | Included | OpenCLIP encoder |
| `scikit-learn` | Installed by `setup_env.sh` | Query/memory models |
| `joblib` | Installed by `setup_env.sh` | Loads trained model bundle |
| `requests` | Installed by `setup_env.sh` | Backend HTTP/Ollama calls |
| `fastapi` | Installed by `setup_env.sh` | Remote API |
| `uvicorn` | Installed by `setup_env.sh` | Runs FastAPI |
| `pydantic` | Installed by `setup_env.sh` | Request/response models |
| `python-multipart` | Installed by `setup_env.sh` | Required for video upload endpoint |
| `pandas` | Installed by setup_env.sh | Required because `qstream_api_server.py` imports pandas |
| PyTorch + CUDA | Must be verified | Prefer a GPU image with CUDA-enabled PyTorch already installed |
| Ollama | Checked, not guaranteed installed | Required for Qwen2.5-VL |
| `qwen2.5vl:7b` | Checked, not automatically guaranteed | Run `ollama pull qwen2.5vl:7b` |
| FFmpeg | Installed remotely by `setup_env.sh` | Also needed locally if using YouTube download/merge |
| `yt-dlp` | Included in backend requirements | Install locally too if using local YouTube download |

### Optional packages currently installed by `setup_env.sh`

`sqlalchemy`, `passlib[bcrypt]`, and `python-jose` are installed by the setup script, but the current local authentication implementation uses Python's built-in `sqlite3`, PBKDF2 hashing, and custom session tokens. They are therefore not required for the current `server.py` runtime, although leaving them installed does not cause a problem.

`imbalanced-learn` is mainly needed for model training/experimentation rather than normal inference with the already exported `qstream_model.joblib` bundle.

The current exported Q-Stream model records its linguistic feature mode as a scikit-learn TF-IDF analyzer fallback, so a separate spaCy model is not required for this specific trained bundle.

---



# 14. Shutdown

When finished:

1. Stop the local server with `Ctrl+C`.
2. Close the SSH-tunnel terminal.
3. Stop the remote API with `Ctrl+C`.
4. Stop Ollama if required.
5. Shut down the rented GPU instance to avoid unnecessary rental charges.

---

## Quick Start Summary

### Remote GPU

```bash
cd /root/autodl-tmp/Q_Stream
bash setup_env.sh        # first time only
pip install pandas       # first time; missing explicit dependency
ollama serve > /tmp/ollama.log 2>&1 &
ollama pull qwen2.5vl:7b # first time only
python qstream_api_server.py
```

### Windows SSH tunnel

```powershell
ssh -N -p [your ssh port] -L 8000:localhost:8000 root@connect.bjb1.seetacloud.com
```

### Windows local server

```powershell
cd "[your path]"
pip install requests     # first time if needed
python server.py --port 5500
```

### Browser

```text
http://localhost:5500/main.html
```

---

**Q-Stream - Query-Aware Streaming Video Understanding**
