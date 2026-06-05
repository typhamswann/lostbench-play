# lostbench-play

A **scalable**, hosted, playable LostBench task server — the real simulator
(`play.py`) wrapped for deployment. Panoramas stream from the public R2 bucket
and render **server-side**, so the browser only receives a JPEG per turn: no
client-side CORS, WebGL, or repo-bloat. Embed it in the LostBench page via
`<iframe>`.

Bundled: `play.py` + `core/` (the simulator) + 5 subset task graphs
(`data/world_graphs/`) listed in `data/tasks.jsonl`. The in-app dropdown lets a
visitor switch between them. ~1 MB; panos fetched on demand.

## Scalability

- **Per-session state.** Each browser (cookie `lb_sid`) gets its own isolated
  sim, kept in-process under a lock with idle-TTL eviction. Concurrent players
  never clobber each other (verified). No global singleton.
- **Concurrency.** Served by gunicorn — 1 worker (so the in-process session
  store is shared) with many threads for simultaneous renders.
- **Horizontal scale.** Run on any container autoscaler. Enable **session
  affinity** so a player's requests return to the instance holding their sim;
  the autoscaler adds instances under load. Tune `LB_SESSION_TTL` /
  `LB_MAX_SESSIONS` (env) to bound per-instance memory.

## Run locally

```bash
pip install -r requirements.txt
WANDERBENCH_PANOS_PUBLIC_URL=https://pub-b410c3932f6242a08d9d3f2d6ed556a8.r2.dev \
  python scripts/play.py --task-id cell_new_00236_easy_02 --compass --map-self --port 8080
# open http://localhost:8080
```

## Deploy (host-agnostic Docker)

The image takes a `PORT` env and serves gunicorn. Works on any container host.

### Google Cloud Run (recommended — autoscale, scale-to-zero, session affinity)
```bash
gcloud run deploy lostbench-play \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --session-affinity \
  --memory 1Gi --cpu 1 --concurrency 12
# -> prints a https://lostbench-play-xxxx.run.app URL
```

### Render
Create a Web Service from this repo/Dockerfile. Enable **session affinity** in
the service settings. Free tier cold-starts (~30 s first hit).

After deploy, put the URL into the LostBench page: edit
`lostbench/index.html`, set the play embed's `data-src` to your deploy URL.

## Config (env)
| var | default | meaning |
|---|---|---|
| `WANDERBENCH_PANOS_PUBLIC_URL` | (set in Dockerfile) | public pano bucket |
| `LB_DEFAULT_TASK` | `cell_new_00236_easy_02` | task shown first |
| `LB_COMPASS` / `LB_MAP_SELF` | `1` / `1` | difficulty toggles |
| `LB_SESSION_TTL` | `1800` | idle-session eviction (s) |
| `LB_MAX_SESSIONS` | `2000` | per-instance hard cap |
