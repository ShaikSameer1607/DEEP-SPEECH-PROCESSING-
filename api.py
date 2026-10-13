"""
FastAPI server for live spoofed-speech attribution.
"""

from pathlib import Path
import tempfile
import time

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Request
)
from fastapi.responses import (
    FileResponse
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from live_test import (
    analyze_audio,
    DEVICE
)


app = FastAPI(
    title="Live Spoofed Speech Attribution API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


ALLOWED_EXTENSIONS = {
    ".wav",
    ".flac"
}


@app.get("/")
def root(request: Request):

    if "text/html" in request.headers.get("accept", ""):
        frontend_path = (
            Path(__file__).resolve().parent
            / "frontend"
            / "index.html"
        )
        if frontend_path.exists():
            return FileResponse(frontend_path)

    return {
        "status": "ok",

        "service":
            "Live Spoofed Speech Attribution",

        "device":
            str(DEVICE),

        "endpoint":
            "/analyze"
    }


@app.get("/ui")
def ui():

    frontend_path = (
        Path(__file__).resolve().parent
        / "frontend"
        / "index.html"
    )

    if frontend_path.exists():
        return FileResponse(frontend_path)

    raise HTTPException(
        status_code=404,
        detail="Frontend index.html not found."
    )


@app.get("/health")
def health():

    return {
        "status": "ok",

        "device":
            str(DEVICE)
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    filename = (
        file.filename
        or "audio.wav"
    )

    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,

            detail=
            "Only WAV and FLAC files "
            "are supported."
        )

    data = await file.read()

    if not data:

        raise HTTPException(
            status_code=400,

            detail=
            "Uploaded file is empty."
        )

    temp_path = None

    try:

        # -----------------------------------------------------
        # Save upload temporarily
        # -----------------------------------------------------
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            tmp.write(data)

            temp_path = Path(
                tmp.name
            )

        # -----------------------------------------------------
        # REAL LIVE INFERENCE
        # -----------------------------------------------------
        start = time.perf_counter()

        result = analyze_audio(
            temp_path
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        result[
            "latency_ms"
        ] = round(
            latency_ms,
            2
        )

        return result

    except Exception as exc:

        print(
            f"/analyze error: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,

            detail=(
                "Live inference failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )
        ) from exc

    finally:

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",

        host="127.0.0.1",

        port=8000,

        reload=False
    )