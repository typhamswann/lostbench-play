# LostBench — single hosted, playable task server.
# Wraps the human-play Flask app (play.py) + the simulator (core/) + a small
# set of task graphs. Panoramas stream from the public R2 bucket at runtime,
# rendered server-side, so the browser only ever receives a base64 JPEG —
# no client-side CORS or WebGL needed.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core   ./core
COPY scripts ./scripts
COPY data   ./data

# Panos come from the public benchmark bucket (server-side fetch + render).
ENV WANDERBENCH_PANOS_PUBLIC_URL=https://pub-b410c3932f6242a08d9d3f2d6ed556a8.r2.dev
ENV PORT=8080
EXPOSE 8080

# Flask dev server (threaded). Fine for a low-traffic demo. The in-env task
# dropdown lets the visitor switch between the bundled tasks; default + compass
# + self-pin on the map are enabled for an inviting first play.
CMD ["sh", "-c", "python scripts/play.py --task-id cell_new_00236_easy_02 --compass --map-self --host 0.0.0.0 --port ${PORT:-8080}"]
