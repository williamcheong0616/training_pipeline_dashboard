from contextlib import asynccontextmanager
import os

import psutil
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.session import init_db
from backend.api.routes import jobs, models, datasets, exports, asr, eval as eval_routes, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Training Pipeline API", version="1.0.0", lifespan=lifespan)

_frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(exports.router)
app.include_router(asr.router)
app.include_router(eval_routes.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/system")
def system_stats():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_memory
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_mb": total // (1024 ** 2),
                "used_mb": allocated // (1024 ** 2),
                "free_mb": (total - allocated) // (1024 ** 2),
            })

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_mb": mem.total // (1024 ** 2),
        "ram_used_mb": mem.used // (1024 ** 2),
        "disk_total_gb": disk.total // (1024 ** 3),
        "disk_used_gb": disk.used // (1024 ** 3),
        "gpu": gpu_info,
        "cuda_available": torch.cuda.is_available(),
    }
