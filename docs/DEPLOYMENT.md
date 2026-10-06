# Throughline — deployment runbook

For the operator: the person who installs, updates and looks after the
Throughline server on the research workstation. It is not a user guide for the
web UI.

**The target rig is not known yet** — operating system, GPU, driver version and
network access are all open questions. Nothing below guesses at them. Every
rig-specific value is a parameter with a conservative default, and every step
that has never actually been executed says so where it appears. Read
[§11 What has actually been verified](#11-what-has-actually-been-verified)
before relying on any of it.

Contents

1. [The one rule](#1-the-one-rule)
2. [Install on Linux](#2-install-on-linux)
3. [Password setup](#3-password-setup)
4. [First start](#4-first-start)
5. [Day-to-day operation](#5-day-to-day-operation)
6. [Access options](#6-access-options)
7. [Update](#7-update)
8. [Rollback](#8-rollback)
9. [Back up and restore the volumes](#9-back-up-and-restore-the-volumes)
10. [GPU — untested pending the rig](#10-gpu--untested-pending-the-rig)
11. [What has actually been verified](#11-what-has-actually-been-verified)
12. [Regenerating the dependency locks](#12-regenerating-the-dependency-locks)
13. [Future: HPC and Apptainer conversion](#13-future-hpc-and-apptainer-conversion)
- [Appendix A — Windows (WSL2)](#appendix-a--windows-wsl2)
- [Appendix B — Configuration reference](#appendix-b--configuration-reference)
- [Appendix C — Security posture](#appendix-c--security-posture)
- [Appendix D — Troubleshooting](#appendix-d--troubleshooting)

---

## 1. The one rule

**The server refuses to start on a network-facing interface unless
`THROUGHLINE_PASSWORD` is set.** It is not a warning; the process exits
(status 2) with a message saying why.

Throughline runs unbounded compute on request — anyone who can reach it can
queue transcript downloads and model inference on the workstation. A loopback
bind (`127.0.0.1`, the default for `panekmodel2 ui`) needs no password because
only that machine can reach it. Anything else needs one.

The Docker image binds `0.0.0.0` *inside* the container, so under Docker the
password is always mandatory — and `docker compose` itself refuses to start
without it, before any container exists.

When the password is set, **every** route requires it: the UI page, its
static assets, `/api/*`, the CSV exports and the API docs. An unknown path
answers `401`, not `404`.

---

## 2. Install on Linux

Linux is the primary target. For Windows see [Appendix A](#appendix-a--windows-wsl2).

### 2.1 Docker (the intended path)

Host prerequisites: Docker Engine with the Compose plugin, and git. Nothing
else — Python, ffmpeg and the model libraries all live in the image.

Install Docker Engine from Docker's own repository for your distribution
(<https://docs.docker.com/engine/install/>), not the distribution's older
`docker.io` package. Then:

```bash
docker version            # the daemon answers ("Server:" section present)
docker compose version    # v2.x
sudo usermod -aG docker "$USER"   # optional: run docker without sudo; log out and in
```

Get the code onto the workstation (wherever you keep it; `/opt/throughline`
is used below as an example):

```bash
git clone <repository-url> /opt/throughline
cd /opt/throughline
```

Make sure the Docker service starts at boot, or `restart: unless-stopped` has
nothing to restart:

```bash
sudo systemctl enable --now docker
```

Disk: budget ~6 GB for the image and the first model download, plus whatever
the transcript cache grows to.

### 2.2 Without Docker (a plain virtualenv)

Useful for sanity-checking a machine, and the fallback if Docker is not
allowed on it. Needs Python 3.10+ (3.12 is what the locks were made with) and
`ffmpeg` if you want the Whisper fallback (`sudo apt install ffmpeg`).

```bash
cd /opt/throughline
python3 -m venv .venv
. .venv/bin/activate
pip install .                   # from the version floors in pyproject.toml
panekmodel2 ui                  # 127.0.0.1:8000, no password needed
```

The committed `requirements.lock` is a **macOS** lock and will refuse to
install on Linux under `--require-hashes` (see [§12](#12-regenerating-the-dependency-locks)).
For a pinned Linux venv, generate a lock on that machine first.

To keep the virtualenv server running permanently without Docker, run it
under systemd (sketch, not executed):

```ini
# /etc/systemd/system/throughline.service
[Service]
User=throughline
WorkingDirectory=/opt/throughline
EnvironmentFile=/opt/throughline/.env
ExecStart=/opt/throughline/.venv/bin/panekmodel2 ui --host 0.0.0.0 --no-open
Restart=on-failure
[Install]
WantedBy=multi-user.target
```

---

## 3. Password setup

Generate it — do not invent one:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**There is no rate limit and no lockout**, so the password's length is the
whole brute-force defence. The server logs a warning at startup if it is under
16 characters (it logs the length, never the value).

Put it in `.env` at the top of the checkout. That file is listed in
`.gitignore` (never committed) and `.dockerignore` (never copied into the
image), and `docker compose` reads it automatically:

```bash
cd /opt/throughline
umask 077
printf 'THROUGHLINE_PASSWORD=%s\n' "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
chmod 600 .env
```

Then give it to the people who need it through a channel that is not plain
email (a password manager share, or in person).

Things that are deliberately true:

- **A blank value counts as unset.** `THROUGHLINE_PASSWORD=` does not create an
  empty password anyone can use; it is treated as absent and a network bind is
  refused.
- **Leading/trailing whitespace is stripped**, so a trailing newline from a
  file does not become part of the password.
- **There is no `--password` command-line flag.** A secret on the command line
  lands in shell history and in `ps` output.
- **The password is never logged**, and it is not part of the application
  settings that `/api/health`, run results and CSV metadata expose.

**Signing in:** the browser shows its own username/password prompt. The
username is ignored (type anything); the password is the value above. Scripts
can send `Authorization: Bearer <password>` instead.

**Rotating it:** edit `.env`, then `docker compose up -d --force-recreate`.
A plain `restart` does **not** pick up a changed environment. Everyone's
browser will prompt again.

---

## 4. First start

```bash
cd /opt/throughline
docker compose up -d --build
```

The first build downloads roughly 1–2 GB of wheels. The first *analysis run*
then downloads ~2 GB of model weights into the `throughline-models` volume;
later restarts reuse them.

Check it came up:

```bash
docker compose ps        # STATUS reaches "(healthy)" within a minute or so
docker compose logs -f   # Ctrl-C stops following, not the server
```

On the workstation itself, open <http://127.0.0.1:8000/> and enter the
password. To reach it from anywhere else, pick an option in [§6](#6-access-options).

Then do one real run on a short video and export a CSV, so that a broken
model download shows up now rather than in front of a user.

---

## 5. Day-to-day operation

```bash
docker compose logs -f --tail 200        # watch
docker compose restart                   # restart (same container, same env)
docker compose up -d --force-recreate    # apply a changed .env
docker compose stop                      # stop; comes back on `start`, not on reboot
docker compose down                      # remove the container; volumes are kept
```

`restart: unless-stopped` brings the container back after a crash or a host
reboot — unless you stopped it yourself.

**Results live only in the server process.** A run's results are discarded
when the container stops, restarts or is recreated. The transcript and model
caches survive (they are on volumes), but results that were not exported to
CSV are gone. Tell users to export before you restart, and check that no run
is in progress (`docker compose logs --tail 50`). `stop_grace_period` is 120 s
to give an in-flight run a chance, but a long batch will not finish in that.

`docker compose down -v` **also deletes the volumes**, i.e. every cached
transcript and model. You almost never want `-v`.

---

## 6. Access options

By default the port is published on the workstation's **loopback only**
(`127.0.0.1:8000`) — nothing else on the network can reach it until you choose
one of the options below. They are in order of preference.

> HTTP Basic credentials are base64-encoded, **not encrypted**. Every option
> below except 6.4 encrypts the traffic for you; 6.4 must not be used on a
> network you do not trust.

### 6.1 SSH tunnel (no changes to anything)

If you can SSH to the workstation:

```bash
# on your laptop
ssh -N -L 8000:127.0.0.1:8000 you@workstation
```

then open <http://127.0.0.1:8000/> on the laptop. Encrypted by SSH, no port
opened. Best for the operator; awkward for a non-technical user.

### 6.2 Tailscale

A private, encrypted network between the workstation and the users' own
devices, with no firewall changes. **Check first whether university policy
allows Tailscale on a university-owned machine** — that is a question for
campus IT, and the answer is not known.

Install Tailscale on the workstation and on each user's device
(<https://tailscale.com/download>), signed into the same tailnet. Then, on the
workstation, either:

**(a) Tailscale Serve — preferred.** Leaves the compose bind on loopback and
gives users an HTTPS address with a real certificate:

```bash
sudo tailscale serve --bg 8000
tailscale serve status        # prints the https://<machine>.<tailnet>.ts.net URL
```

Requires MagicDNS and HTTPS certificates enabled in the tailnet admin console.

**(b) Publish on the Tailscale interface only.** Put the workstation's
Tailscale address (`tailscale ip -4`, a `100.x.y.z` address) in `.env`:

```
THROUGHLINE_BIND_ADDR=100.x.y.z
```

then `docker compose up -d --force-recreate`. Traffic is encrypted by
WireGuard; users browse to `http://100.x.y.z:8000/`. If Tailscale is not yet
up when Docker starts after a reboot, the publish fails — (a) avoids that.

Either way, use Tailscale's access controls to limit which devices can reach
the workstation.

### 6.3 Campus IT

If the server needs to be reachable on the university network without
software on every client, it is a request to campus IT. Questions to bring:

- Can the workstation get a stable hostname/IP, and is it behind the campus
  firewall (reachable on campus only) or the campus VPN?
- Can they put it behind a TLS-terminating reverse proxy with a university
  certificate? (Then bind `THROUGHLINE_BIND_ADDR` to the address the proxy
  reaches, and the password still applies behind the proxy.)
- Do they require their own authentication (SSO) in front? The password gate
  stays on either way; SSO in front is extra, not a replacement.
- Are long-running, high-CPU/GPU services allowed on that machine?

### 6.4 Plain LAN publish (trusted networks only)

```
# .env
THROUGHLINE_BIND_ADDR=0.0.0.0
```

then `docker compose up -d --force-recreate`. The password is enforced, but
it crosses the network readable by anything on the path. Only on a network
you control.

---

## 7. Update

```bash
cd /opt/throughline
# 0. tell users; make sure nothing is running; results not exported are lost
git log -1 --format='%h %s'                   # note the current version
docker tag throughline:local throughline:previous   # keep the running image for §8
git pull                                      # or: git fetch && git checkout <tag>
docker compose up -d --build
docker compose ps                             # wait for "(healthy)"
```

Then do one short run as in §4. If the update changed dependencies, the build
takes longer; the volumes are untouched.

---

## 8. Rollback

**Fast path — the image you tagged in §7:**

```bash
docker tag throughline:previous throughline:local
docker compose up -d --no-build --force-recreate
```

**From source — any earlier version:**

```bash
git log --oneline -20                # find the version to return to
git checkout <commit-or-tag>
docker compose up -d --build
```

Return to the newest code later with `git checkout main && git pull`.

The volumes only hold caches. If an older version rejects the newer cache
format, delete the transcript cache — it is rebuilt on demand:

```bash
docker compose down
docker volume rm throughline-transcripts
docker compose up -d
```

---

## 9. Back up and restore the volumes

| Volume | Holds | Lose it and… |
|---|---|---|
| `throughline-transcripts` | fetched transcripts, per-chunk results | re-fetched on demand (slow; YouTube may rate-limit) |
| `throughline-models` | Hugging Face / Whisper / NLTK model files | ~2 GB re-downloaded on next run |
| `throughline-exports` | files CLI commands wrote inside the container | gone |

Only `throughline-transcripts` and `throughline-exports` are worth backing up;
the models are re-downloadable. The web UI's CSV exports download to the
user's browser and are not on any volume.

Back up (stop first, so nothing is mid-write; not executed — see §11):

```bash
mkdir -p ~/throughline-backups && cd ~/throughline-backups
docker compose -f /opt/throughline/docker-compose.yml stop
for v in throughline-transcripts throughline-exports; do
  docker run --rm --user "$(id -u):$(id -g)" --entrypoint tar \
    -v "$v":/data:ro -v "$PWD":/backup \
    throughline:local czf "/backup/$v-$(date +%F).tar.gz" -C /data .
done
docker compose -f /opt/throughline/docker-compose.yml start
```

Restore into an empty volume:

```bash
docker compose -f /opt/throughline/docker-compose.yml down
docker volume rm throughline-transcripts
C=/home/throughline/.panekmodel2_cache      # the volume's real mount point
docker run --rm --entrypoint tar \
  -v throughline-transcripts:"$C" -v "$PWD":/backup:ro \
  throughline:local xzf /backup/throughline-transcripts-YYYY-MM-DD.tar.gz -C "$C"
docker compose -f /opt/throughline/docker-compose.yml up -d
```

Mount the volume at its **real** path (for exports: `/home/throughline/exports`)
and run as the image's default user, as above. Docker seeds a new, empty
volume with the ownership of the image directory it is mounted over; mounted
anywhere else (say `/data`), it is created root-owned and the server — which
runs as uid 10001 — cannot write to it.

Also back up `.env` (it holds the password) — separately, somewhere private.

---

## 10. GPU — untested pending the rig

**UNTESTED-PENDING-RIG. Nothing in the GPU path has been executed.** No NVIDIA
GPU, NVIDIA Container Toolkit or running Docker daemon existed where this was
written, and the workstation's GPU is not known. The CPU baseline works
without any of this.

1. Verify the host on its own, before involving this project:

   ```bash
   nvidia-smi    # note the "CUDA Version" — the driver's maximum
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

   If the second command does not print a GPU table, fix the NVIDIA Container
   Toolkit installation first; nothing here will help.

2. Choose a torch build the driver supports, in `.env`. `cu124` is a
   placeholder, not a recommendation:

   ```
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
   ```

3. Build and start with the override:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
   ```

4. Confirm the container really sees the device — do not infer it from the
   absence of an error:

   ```bash
   docker compose exec throughline python -c \
     "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
   ```

   `False` with no error is the usual symptom of a driver/wheel mismatch.
   Expect the image to grow by 2–3 GB.

Every later `docker compose` command must then carry both `-f` flags, or it
silently falls back to the CPU configuration.

---

## 11. What has actually been verified

Distinguishing these matters more than this document looking finished.

**Executed and passing** (macOS/arm64, Python 3.12, no Docker daemon):

- The password gate and the bind refusal, by automated test: every registered
  route including the static mount, `/redoc` and `/openapi.json`; both
  schemes; malformed credentials; the blank-password case; the CLI exit code;
  and real-socket tests through a running uvicorn server.
- No credential appears in any log record, checked over a real connection with
  uvicorn at its most verbose (`trace`) level and every logger captured.
- Mutation testing of the auth enforcement (each mutation verified applied,
  then the suite run); the results are in the evidence bundle
  `artifacts/implementation/packaging.evidence.yaml`.
- `requirements-dev.lock` installs under `--require-hashes` into a fresh venv,
  and the full fast test suite passes in that venv.
- The healthcheck script against a real gated server.
- Both compose files, as **Docker's own parser** resolves them
  (`docker compose config`, Compose v2.34 — client-side, no daemon needed):
  it refuses an unset or blank password, publishes on `127.0.0.1`, mounts the
  three named volumes, and the GPU override switches only the torch index,
  `CUDA` and the device reservation.

**Not executed — treat as unverified:**

- **The Docker image has never been built**, so nothing about its contents is
  verified: whether it builds, the non-root user and volume ownership, NLTK
  and model downloads landing in the volume, and the healthcheck inside the
  container. These need a machine with a running Docker daemon. The tests
  check the Dockerfile's *text* only.
- **The entire GPU path** (§10) and `docker-compose.gpu.yml`.
- **`scripts/lock.sh --docker`**, and therefore a reproducible (pinned) image.
- **Linux of any kind**, and Windows/WSL2. No step has run on the operating
  system the workstation most likely uses.
- The backup/restore commands (§9), Tailscale (§6.2), the systemd sketch
  (§2.2) and the Apptainer notes (§13).
- Behaviour under real model workloads, concurrent users, or large batches.

First thing to do on the real rig: §4, then one real run, then a §9 backup and
restore of the transcripts volume — and update this section.

---

## 12. Regenerating the dependency locks

`requirements.lock` (runtime) and `requirements-dev.lock` (runtime + test
tools) are exact pins with sha256 hashes, generated by pip-tools.
`pyproject.toml` is the input and the single list of dependencies — it plays
the part a `requirements.in` usually plays, so there is no second list to
drift from it. Regenerate after changing a dependency in `pyproject.toml`
(a test fails if a declared dependency is missing from the lock):

```bash
python3.12 -m venv /tmp/lockenv
/tmp/lockenv/bin/pip install 'pip==24.2' 'pip-tools==7.4.1'
PYTHON=/tmp/lockenv/bin/python ./scripts/lock.sh
```

The pip-tools version is pinned on purpose: 7.6.1 builds the PyPI JSON-API
URL wrongly and silently falls back to downloading every wheel of every
platform just to hash it (it passed 22 GB of torch wheels before being
stopped). `lock.sh` refuses any other version unless `LOCK_ANY_PIP_TOOLS=1`.

Then prove the lock, rather than assuming it:

```bash
python3.12 -m venv /tmp/lockcheck
/tmp/lockcheck/bin/pip install --require-hashes -r requirements-dev.lock
/tmp/lockcheck/bin/pip install --no-deps --no-build-isolation -e .
/tmp/lockcheck/bin/python -m pytest -m "not slow"
```

**A lock is only valid for the platform and Python version that produced it.**
The committed locks were resolved on macOS/arm64 with Python 3.12. On Linux,
torch and Whisper declare extra dependencies (nvidia-*, triton) that a macOS
resolution leaves out, so `pip install --require-hashes -r requirements.lock`
fails there — loudly, not by installing something unpinned. That is why the
Docker image installs from the `pyproject.toml` floors by default.

For a pinned image, resolve inside the image's base and build against it
(not executed — needs a Docker daemon):

```bash
./scripts/lock.sh --docker          # writes requirements-docker.lock
REQUIREMENTS_FILE=requirements-docker.lock docker compose build
```

---

## 13. Future: HPC and Apptainer conversion

Not done and not tested; notes for when a cluster becomes the target.
Clusters run Apptainer (formerly Singularity), not Docker, and run *jobs*,
not permanent services.

- **Convert the image** on a machine with Docker, then copy the `.sif` over:

  ```bash
  docker save throughline:local -o throughline.tar
  apptainer build throughline.sif docker-archive://throughline.tar
  ```

- **No daemon, no restart policy.** A server runs inside a scheduler job
  (Slurm) for that job's wall-time. "Running permanently" becomes "submit a job
  when needed".
- **Runs as you, read-only image.** Apptainer binds your home directory by
  default, so the caches land in your real `$HOME`, which on clusters usually
  has a small quota. Point them at scratch:
  `--env HF_HOME=/scratch/$USER/hf,NLTK_DATA=/scratch/$USER/nltk` and bind a
  scratch directory for `~/.panekmodel2_cache`.
- **Compute nodes often have no internet.** Pre-download the models into the
  scratch cache from a login node before submitting.
- **GPUs:** `apptainer run --nv throughline.sif ...` passes the node's GPU
  through; the torch build must still match the node's driver (§10).
- **Access:** a server on a compute node is reached by an SSH tunnel through
  the login node (§6.1). The bind rule still applies — a non-loopback bind
  needs `THROUGHLINE_PASSWORD`, passed with `--env` or an env file, never on
  the command line of a shared node where `ps` is visible to other users.

---

## Appendix A — Windows (WSL2)

Not executed. Two routes, both running the Linux image under WSL2:

1. **Docker Desktop with the WSL2 backend** (simplest). Install WSL2
   (`wsl --install` in an administrator PowerShell, then reboot) and Docker
   Desktop. Licensing: Docker Desktop requires a paid subscription for larger
   organisations — check whether that applies to the university.
2. **Docker Engine inside a WSL2 Ubuntu distribution**, installed as in §2.1.
   Avoids Docker Desktop, but you must arrange for WSL and Docker to start at
   boot yourself.

Then, inside the Ubuntu (WSL) shell, follow §2–§4 unchanged. Notes:

- Clone into the Linux filesystem (`~/throughline`), **not** under `/mnt/c/`:
  builds and the caches are far slower across the Windows boundary.
- A port published on `127.0.0.1` is reachable from Windows at
  <http://localhost:8000/>.
- **"Permanently running" depends on Windows.** `restart: unless-stopped` only
  helps once Docker is running: set Docker Desktop to start at sign-in, and
  stop the machine sleeping (Settings → System → Power).
- GPU under WSL2 needs the NVIDIA *Windows* driver with WSL support; do not
  install a Linux NVIDIA driver inside WSL. Then §10 applies.
- Tailscale (§6.2): install it on Windows itself.

---

## Appendix B — Configuration reference

Set in `.env` next to `docker-compose.yml`. Build-time values need
`docker compose up -d --build`; the rest need `--force-recreate`.

| Variable | Default | Notes |
|---|---|---|
| `THROUGHLINE_PASSWORD` | *(required)* | Compose refuses to start without it (§3) |
| `THROUGHLINE_BIND_ADDR` | `127.0.0.1` | Host interface the port is published on (§6) |
| `THROUGHLINE_PORT` | `8000` | Host port |
| `YOUTUBE_API_KEY` | *(empty)* | Metadata only; captions never need it |
| `HF_TOKEN` | *(empty)* | Only for gated Hugging Face models |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Topic-model embeddings |
| `SENTIMENT_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 3-class |
| `TOPIC_GRANULARITY` | `standard` | `coarse` / `standard` / `fine` |
| `TOPIC_REDUCE_TO` | `10` | `0` disables reduction |
| `CHUNK_MAX_SECONDS` | `60` | |
| `USE_WHISPER_FALLBACK` | `false` | Transcribe audio when a video has no captions; slow |
| `WHISPER_MODEL` | `small` | |
| `CUDA` | `false` | Set by the GPU override |
| `TORCH_INDEX_URL` | CPU wheel index | Build-time (§10) |
| `REQUIREMENTS_FILE` | *(empty)* | Build-time; a Linux lock for a pinned image (§12) |
| `GPU_COUNT` | `1` | GPU override only |

---

## Appendix C — Security posture

What the gate does: every request — page, static asset, API, export, docs —
needs the password, by HTTP Basic (browsers) or Bearer (scripts, the
healthcheck), compared in constant time. With no password, the server refuses
any non-loopback bind.

What it does **not** do — plan around these:

- **No transport encryption** of its own. Use §6.1–6.3.
- **No rate limiting or lockout.** Entropy is the only brute-force defence,
  hence §3's generated password.
- **One shared password.** No accounts, no per-user audit trail, no logout — a
  browser keeps the credential until it is closed. Revoking one person means
  rotating it for everyone.
- **Failed sign-ins are not logged.** Deliberate (an unauthenticated stranger
  must not be able to drive log volume, and the credential must never reach a
  log), with the consequence that a guessing attack leaves no trace in these
  logs.
- **Compute is still unbounded once signed in.** Resource limits are commented
  out in `docker-compose.yml` rather than guessed; set them once the rig's
  memory is known.
- **`docker compose config` prints the password in plain text.** Never paste
  its output anywhere.

---

## Appendix D — Troubleshooting

**`THROUGHLINE_PASSWORD is required`** from `docker compose` — Compose refused
before building anything. Create `.env` per §3.

**`Refused to start.` / exit status 2** — the server's own bind refusal: the
password did not reach the process. Under systemd, check `EnvironmentFile`.

**Container restarting in a loop** — `docker compose logs --tail 50`.

**`(unhealthy)` but the UI works** — the healthcheck's credential disagrees
with the server's, usually after editing `.env` and only restarting:
`docker compose up -d --force-recreate`.

**Browser keeps re-prompting** — wrong password, or the password was rotated.
Close all windows of that browser to clear the stored credential.

**First run very slow** — model weights downloading (~2 GB). Only the first
run pays it, as long as the `throughline-models` volume survives.

**A video is skipped with "No caption track"** — expected for videos without
captions. `USE_WHISPER_FALLBACK=true` transcribes the audio instead (much
slower; ffmpeg is already in the image).

**Nothing at `http://<workstation-ip>:8000`** — the port is published on
loopback by default. See §6.

**Permission denied writing the cache** — a volume was created or restored
with the wrong owner. The server runs as uid 10001; see the restore command in
§9.
