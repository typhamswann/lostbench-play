# LostBench — scalable, hosted playable-task server.
# The human-play Flask app (play.py) + the simulator (core/) + a small set of
# task graphs. Panoramas stream from the public R2 bucket and render
# server-side, so the browser only receives a base64 JPEG per turn — no
# client-side CORS or WebGL.
#
# Scalability: sim state is per browser session (cookie), not a global
# singleton, so concurrent players are isolated. Served by gunicorn (single
# worker so in-process sessions are shared, many threads for concurrency).
# Host on any autoscaler (Cloud Run, Render, etc.) with SESSION AFFINITY so a
# player's requests return to the instance holding their sim; the autoscaler
# adds instances under load.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core    ./core
COPY scripts ./scripts
COPY data    ./data

ENV WANDERBENCH_PANOS_PUBLIC_URL=https://pub-b410c3932f6242a08d9d3f2d6ed556a8.r2.dev \
    LB_DEFAULT_TASK=cell_new_00236_easy_02 \
    LB_COMPASS=1 \
    LB_MAP_SELF=1 \
    PORT=8080
EXPOSE 8080

# 1 worker (shared in-process session store) + threads for concurrency.
# Scale OUT with more instances behind a session-affinity load balancer.
CMD ["sh", "-c", "gunicorn --chdir scripts play:app -w 1 --threads 12 -b 0.0.0.0:${PORT:-8080} --timeout 120"]
