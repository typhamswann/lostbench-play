# lostbench-play

A single hosted, **playable** LostBench task server — the real simulator
(`play.py`) wrapped for deployment. Panoramas stream from the public R2 bucket
and are rendered **server-side**, so the browser only receives a JPEG per turn:
no client-side CORS, WebGL, or repo-bloat. Embed it in the LostBench page via
`<iframe>`.

Bundled: `play.py` + `core/` (the simulator) + 5 subset task graphs
(`data/world_graphs/`) listed in `data/tasks.jsonl`. The in-app dropdown lets a
visitor switch between them. ~1 MB total; panos are fetched on demand.

## Run locally

```bash
pip install -r requirements.txt
WANDERBENCH_PANOS_PUBLIC_URL=https://pub-b410c3932f6242a08d9d3f2d6ed556a8.r2.dev \
  python scripts/play.py --task-id cell_new_00236_easy_02 --compass --map-self --port 8080
# open http://localhost:8080
```

## Deploy (Fly.io — recommended, always-on small VM)

```bash
# one-time: brew install flyctl && fly auth login
fly launch --no-deploy --copy-config --name lostbench-play   # accept the bundled fly.toml
fly deploy
# -> https://lostbench-play.fly.dev
```

That URL goes straight into the LostBench page's "Play the Benchmark" iframe.

### Alternative: Render / Railway
Both deploy this Dockerfile directly. Render's free tier cold-starts (~30 s
first hit); Fly's `auto_stop_machines` also cold-starts but faster. For a demo
either is fine.

## Known limitation — single shared session

`play.py` keeps **one** global sim in module state, so concurrent visitors
would share (and clobber) each other's position. Fine for a low-traffic
portfolio demo; if it needs true multi-user, the fix is per-Flask-session sims
(key the sim dict by a session cookie). Flagged, not yet done.
