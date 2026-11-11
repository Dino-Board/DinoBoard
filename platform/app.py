"""DinoBoard web platform entry point."""
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from game_service.routes import router as game_router, replay_router  # noqa: E402
from game_service.pipeline import shutdown_executors  # noqa: E402
from game_service.sessions import (  # noqa: E402
    start_session_reaper,
    stop_session_reaper,
)
from ai_service.routes import router as ai_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_session_reaper()
    yield
    stop_session_reaper()
    shutdown_executors()


app = FastAPI(title="DinoBoard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dev mode: disable browser caching for all responses so CSS / JS / HTML
# changes show up on a normal refresh. Without this, browsers hang on to
# stale static files across edits and users have to Cmd+Shift+R after
# every tweak. Switch this off (or make it conditional on an env var) if
# you ever deploy this to actual production.
@app.middleware("http")
async def no_cache(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

app.include_router(game_router)
app.include_router(replay_router)
app.include_router(ai_router)

# --- Static file serving ---

static_dir = PROJECT_ROOT / "platform" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

games_dir = PROJECT_ROOT / "games"


# Append `?v=<mtime>` to every same-origin CSS/JS reference in an index.html
# so aggressive ISP/browser/webview caches (common on Chinese mobile browsers
# that ignore Cache-Control) treat each edit as a fresh URL and can't serve
# a stale copy. Only relative refs are rewritten — absolute URLs, CDN refs,
# and already-versioned URLs are left alone.
_ASSET_REF_RE = re.compile(
    r'(?P<attr>href|src)="(?P<url>[^"]+\.(?:css|js))"'
)


def _rewrite_asset_refs(html: str, web_dir: Path) -> str:
    def repl(m: re.Match) -> str:
        url = m.group("url")
        if "?" in url or "://" in url or url.startswith("/"):
            return m.group(0)
        asset = web_dir / url
        if not asset.exists():
            return m.group(0)
        mtime = int(asset.stat().st_mtime)
        return f'{m.group("attr")}="{url}?v={mtime}"'
    return _ASSET_REF_RE.sub(repl, html)


def _make_game_index_handler(web_dir: Path):
    index_path = web_dir / "index.html"

    def handler() -> HTMLResponse:
        html = index_path.read_text(encoding="utf-8")
        return HTMLResponse(_rewrite_asset_refs(html, web_dir))
    return handler


for game_dir in sorted(games_dir.iterdir()):
    web_dir = game_dir / "web"
    if web_dir.exists() and (web_dir / "index.html").exists():
        game_name = game_dir.name
        # Custom index handler that injects cache-busting query strings.
        # Must be registered before the StaticFiles mount, otherwise the
        # mount shadows the bare-path route for `/games/<game>/`.
        app.add_api_route(
            f"/games/{game_name}/",
            _make_game_index_handler(web_dir),
            methods=["GET"],
            include_in_schema=False,
        )
        app.add_api_route(
            f"/games/{game_name}",
            _make_game_index_handler(web_dir),
            methods=["GET"],
            include_in_schema=False,
        )
        app.mount(
            f"/games/{game_name}",
            StaticFiles(directory=str(web_dir), html=True),
            name=f"game_{game_name}",
        )


@app.get("/")
def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>DinoBoard</h1><p>No index.html found.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
