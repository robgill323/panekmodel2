"""Container healthcheck: is the server up and answering?

Run by HEALTHCHECK in the image. Exits 0 when /api/health returns 200.

The password is read from the environment inside this process rather than
passed as an argument, so it never appears in ``ps`` output, in ``docker
inspect``, or in a shell history. That is the reason the gate accepts a Bearer
token at all — a healthcheck cannot answer a browser's Basic auth prompt.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("THROUGHLINE_HEALTHCHECK_URL", "http://127.0.0.1:8000/api/health")
TIMEOUT_S = 10


def main() -> int:
    request = urllib.request.Request(URL, method="GET")

    # Absent or blank means the server is running ungated, so no header is
    # sent — the same "blank is unset" rule the server itself applies.
    password = (os.environ.get("THROUGHLINE_PASSWORD") or "").strip()
    if password:
        request.add_header("Authorization", f"Bearer {password}")

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            if response.status != 200:
                print(f"unhealthy: HTTP {response.status}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as exc:
        # 401 here means the container's password and the server's disagree,
        # which is worth saying plainly — it looks identical to "down".
        detail = " (healthcheck credentials rejected)" if exc.code == 401 else ""
        print(f"unhealthy: HTTP {exc.code}{detail}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — any failure is "not healthy"
        print(f"unhealthy: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
