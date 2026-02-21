"""Minimal FastAPI app for the concussion sideline demo.

Use this instead of backend.main when you only need:
- POST /upload-clip
- WS /live-stream
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.concussion_pipeline import router as concussion_router


app = FastAPI(title="BRACE Concussion Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(concussion_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
