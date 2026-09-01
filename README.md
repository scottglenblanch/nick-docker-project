# Nick Docker Project

Local Docker stack for Open WebUI + SearXNG web search.

## Prerequisites

- Docker Desktop with Compose support
- Git Bash (or another shell)

## First-time setup

1. Copy `.env.example` to `.env`.
2. Set a strong random secret for `SEARXNG_SECRET_KEY`.

Example (Git Bash):

```bash
cp .env.example .env
openssl rand -hex 32
```

Then paste the generated value into `.env`:

```env
SEARXNG_SECRET_KEY=your_generated_hex_value
```

## Start services

```bash
docker compose up -d
```

Open:

- Open WebUI: http://localhost:3001
- SearXNG: http://localhost:8080

## Verify status

```bash
docker compose ps
```

## Security notes

- Keep `.env` private. It is ignored by git.
- `searxng-config/settings.yml` intentionally keeps `CHANGE_ME_LOCAL_SECRET` as a placeholder.
- At container startup, Docker injects the real secret from `.env` into a runtime copy.
- If a secret was previously committed or shared, rotate it (generate a new one in `.env`).

## Web search notes

- Open WebUI is configured to use SearXNG via internal Docker network.
- If a chat does not use web search, start a new chat after any restart and retry.
