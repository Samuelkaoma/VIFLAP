# Multi-stage build.
#
# The runtime stage carries no compiler and no build tooling. A forensic service
# holding evidence about identified individuals should not also ship the means to
# compile whatever an attacker manages to write to disk.

FROM python:3.12-slim AS build

WORKDIR /build

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY viflap ./viflap

RUN python -m pip install --no-cache-dir --upgrade pip build \
 && python -m build --wheel --outdir /wheels

# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

# ffmpeg supplies the reference AMR-NB codec. Where the build lacks an AMR
# encoder — many do, for licensing reasons — the system falls back to the
# parametric CELP model and records that it did so on every degraded signal, so
# results from the two are never silently pooled.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Unprivileged, with a home the process can write to and an application
# directory it cannot.
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 viflap

WORKDIR /app
COPY --from=build /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/*.whl "uvicorn[standard]>=0.27" \
 && rm -rf /wheels

# The audit log and model artefacts live on a mounted volume, not in the image.
# An audit log inside a container filesystem is destroyed by a redeploy, which
# is indistinguishable from an audit log destroyed deliberately.
RUN mkdir -p /var/lib/viflap/audit /var/lib/viflap/models \
 && chown -R viflap:viflap /var/lib/viflap
VOLUME ["/var/lib/viflap"]

USER viflap

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIFLAP_AUDIT_LOG=/var/lib/viflap/audit/audit.jsonl \
    VIFLAP_MODEL_DIR=/var/lib/viflap/models

EXPOSE 8000

# Health includes the audit chain's integrity, so an orchestrator restarting an
# unhealthy container is also reacting to a broken chain.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,json,sys; \
b=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4)); \
sys.exit(0 if b.get('status')=='ok' and b.get('audit_chain_intact') else 1)"

CMD ["uvicorn", "--factory", \
     "viflap.interfaces.bootstrap:build_demonstration_container", \
     "--host", "0.0.0.0", "--port", "8000"]
