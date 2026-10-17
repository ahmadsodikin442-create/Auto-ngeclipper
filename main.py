from pathlib import Path
import shutil, subprocess, uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)
app = FastAPI(title="ClipForge AI - Starter")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))
ALLOWED = {".mp4", ".mov", ".webm", ".m4v"}


def run_ffmpeg(args):
    try:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg belum terpasang. Instal FFmpeg lalu pastikan perintah ffmpeg tersedia di PATH.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("FFmpeg gagal memproses video. Pastikan file video valid dan ruang disk cukup.") from exc


def video_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def score_moments(path: Path, duration: float, clip_len: int):
    """Heuristic: sample frames; score visual change/motion. Not semantic AI."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError("Tidak dapat membaca frame video.")
    # Sample roughly twice per second at small resolution.
    step = max(1, int(fps / 2))
    samples = []
    prev = None
    for frame_no in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is None:
            change = 0.0
        else:
            change = float(np.mean(cv2.absdiff(gray, prev))) / 255.0
        # Timestamp in seconds. Motion/visual change is only a rough highlight signal.
        samples.append((frame_no / fps, change))
        prev = gray
    cap.release()
    if not samples:
        raise RuntimeError("Tidak ada frame yang bisa dianalisis.")

    # Candidate starts every 5 seconds, score by average visual change within window.
    starts = list(np.arange(0, max(0.01, duration - min(clip_len, duration)) + 0.1, 5.0))
    ranked = []
    for start in starts:
        end = min(duration, start + clip_len)
        vals = [v for t, v in samples if start <= t < end]
        score = float(np.mean(vals)) if vals else 0.0
        ranked.append((score, float(start), float(end)))
    ranked.sort(reverse=True)
    selected = []
    for score, start, end in ranked:
        if all(abs(start - existing[1]) >= clip_len * 0.7 for existing in selected):
            selected.append((score, start, end))
        if len(selected) >= 5:
            break
    if not selected:
        selected = [(0.0, 0.0, min(duration, clip_len))]
    selected.sort(key=lambda item: item[1])
    return selected


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, video: UploadFile = File(...), clip_seconds: int = Form(30), clip_count: int = Form(3)):
    ext = Path(video.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Format file harus MP4, MOV, WebM, atau M4V.")
    clip_seconds = max(10, min(90, clip_seconds))
    clip_count = max(1, min(5, clip_count))
    job = uuid.uuid4().hex[:10]
    source = UPLOADS / f"{job}{ext}"
    # Limit uploads to 250 MiB for the starter deployment.
    max_bytes = 250 * 1024 * 1024
    total_bytes = 0
    try:
        with source.open("wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    source.unlink(missing_ok=True)
                    raise HTTPException(413, "Ukuran video maksimal 250 MB untuk versi awal.")
                out.write(chunk)
    finally:
        await video.close()
    try:
        duration = video_duration(source)
        if duration < 2:
            raise RuntimeError("Video terlalu pendek.")
        candidates = score_moments(source, duration, clip_seconds)
        # Keep top scoring candidates up to requested count.
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[:clip_count]
        results = []
        for idx, (score, start, end) in enumerate(chosen, 1):
            output = OUTPUTS / f"{job}_clip_{idx}.mp4"
            # Center-crop to portrait 9:16, then scale to 1080x1920.
            vf = "crop='if(gt(iw/ih,9/16),ih*9/16,iw)':'if(gt(iw/ih,9/16),ih,iw*16/9)',scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"
            run_ffmpeg(["-ss", f"{start:.2f}", "-i", str(source), "-t", f"{max(1.0, end-start):.2f}", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)])
            results.append({"name": output.name, "url": f"/download/{output.name}", "start": round(start, 1), "end": round(end, 1), "score": round(score * 100, 1)})
        return templates.TemplateResponse("results.html", {"request": request, "results": results, "duration": round(duration, 1)})
    except Exception as exc:
        return templates.TemplateResponse("error.html", {"request": request, "message": str(exc)}, status_code=500)
    finally:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/download/{filename}")
def download(filename: str):
    # Restrict downloads to generated MP4 files within outputs.
    if Path(filename).name != filename or not filename.endswith(".mp4"):
        raise HTTPException(404, "File tidak ditemukan")
    path = OUTPUTS / filename
    if not path.exists():
        raise HTTPException(404, "File tidak ditemukan")
    return FileResponse(path, media_type="video/mp4", filename=filename)
