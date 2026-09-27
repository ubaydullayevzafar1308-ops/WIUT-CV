# Deployment: website + demo API

One Ubuntu VM with Docker (compose plugin). Caddy serves `https://$SITE_DOMAIN` with an automatic certificate.
`/api/*` goes to the demo API (`demo/api.py`, CPU, light mode), and everything else goes to the website
(https://github.com/86Hoji/Asila_front_elim_task).

The model repository is private and the sample videos are under NDA. So the code is copied to the server from a
local checkout, as tracked files only (no `.git`, `samples/`, `cache/`, `outputs/`, `configs/local/`, `docs/`):

```bash
git ls-files -z | rsync -a --from0 --files-from=- ./ asila@<server>:/opt/asila/model/
```

On the server:

```bash
git clone https://github.com/86Hoji/Asila_front_elim_task.git /opt/asila/site
cd /opt/asila/model/deploy
docker compose up -d --build              # SITE_DOMAIN and SITE_DIR can be overridden in the environment
curl -fsS https://81-26-183-186.sslip.io/api/health
```

| Service | Image | Port | Notes |
| --- | --- | --- | --- |
| api | `deploy/api.Dockerfile` | 8000 (internal) | CPU torch; healthcheck on `/api/health` |
| site | the site's Dockerfile, built with `VITE_USE_MOCK=false`, `VITE_API_BASE=` | 3000 (internal) | the demo API on the same domain |
| caddy | `caddy:2` | 80, 443 | HTTPS, 310 MiB request body limit |

All three services use `restart: unless-stopped`, and Docker starts on boot.
