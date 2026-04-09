"""FastAPI application for gREV."""

from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    try:
        from openenv_core.env_server.http_server import create_app
    except ImportError:
        create_app = None

try:
    from grev.models import GrevAction, GrevObservation
    from grev.env import gREVEnv
except ImportError:
    from models import GrevAction, GrevObservation
    from grev.env import gREVEnv

from fastapi.responses import JSONResponse

app = create_app(
    gREVEnv,
    GrevAction,
    GrevObservation,
    env_name="grev",
    max_concurrent_envs=1,
)


@app.get("/")
async def root_health():
    return JSONResponse(content={"status": "healthy", "message": "gREV is alive!"})


@app.get("/health")
async def explicit_health():
    return JSONResponse(content={"status": "healthy"})


def main(host: str = "0.0.0.0", port: int = 7860):
    """Run the server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
