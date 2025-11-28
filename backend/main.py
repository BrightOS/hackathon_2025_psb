from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

from routers.ping import api_router as ping_api_router
from routers.routes import api_router as api_router

from config import settings


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ping_api_router)
app.include_router(api_router)

if __name__ == "__main__":
    run(
        "main:app",
        host=settings.HOST,
        port=int(settings.PORT),
        log_level="debug",
        timeout_keep_alive=60
    )
