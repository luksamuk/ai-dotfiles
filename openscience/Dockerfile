# Dockerfile for OpenScience — containerized deploy
#
# OpenScience is a Bun-compiled native binary (not a Node.js app).
# The release ships pre-compiled binaries per platform, so the image
# just needs to fetch the binary and run it as a non-root user.
#
# The server ALWAYS binds to 127.0.0.1 (hardcoded, no --hostname flag).
# We use network_mode: host in docker-compose to expose it on the host.
#
# Security: runs as non-root user 'openscience' (UID 1001).
# No database — state is all on-disk (JSON + files under XDG dirs).

FROM debian:bookworm-slim AS base

# Runtime deps: curl for healthchecks, ca-certificates for HTTPS API calls,
# git for repo cloning (agent tool), basic shell tools the agent runtime needs.
# libstdc++ is required by the Bun-compiled binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: fetch the binary ---
FROM base AS fetch

ARG OPENSCIENCE_VERSION=1.3.2
ARG TARGETARCH

# Map docker arch to openscience release arch
RUN case "$TARGETARCH" in \
      amd64)  OS_ARCH="linux-x64" ;; \
      arm64)  OS_ARCH="linux-arm64" ;; \
      *)      echo "Unsupported arch: $TARGETARCH" && exit 1 ;; \
    esac && \
    curl -fsSL \
      "https://github.com/synthetic-sciences/openscience/releases/download/v${OPENSCIENCE_VERSION}/openscience-${OS_ARCH}.tar.gz" \
      -o /tmp/openscience.tar.gz && \
    # Verify checksum if available (releases before checksums.txt skip silently) \
    SHA=$(curl -fsSL \
      "https://github.com/synthetic-sciences/openscience/releases/download/v${OPENSCIENCE_VERSION}/checksums.txt" \
      2>/dev/null | awk -v f="openscience-${OS_ARCH}.tar.gz" '$2 == f {print $1}') && \
    if [ -n "$SHA" ]; then \
      ACTUAL=$(sha256sum /tmp/openscience.tar.gz | awk '{print $1}'); \
      if [ "$ACTUAL" != "$SHA" ]; then \
        echo "Checksum mismatch: expected $SHA, got $ACTUAL" && exit 1; \
      fi; \
      echo "Checksum verified"; \
    fi && \
    tar -xzf /tmp/openscience.tar.gz -C /tmp && \
    chmod +x /tmp/openscience && \
    rm /tmp/openscience.tar.gz

# --- Stage 3: runtime ---
FROM base AS runtime

# Create non-root user (UID 1001, no login shell)
RUN groupadd --gid 1001 openscience && \
    useradd --uid 1001 --gid 1001 --create-home --shell /usr/sbin/nologin openscience

# Copy binary from fetch stage
COPY --from=fetch --chown=openscience:openscience /tmp/openscience /usr/local/bin/openscience

# Create data directories with correct ownership
# OpenScience uses XDG dirs: ~/.config/openscience, ~/.local/share/openscience, ~/.cache/openscience
# We redirect all state to /data via env vars for portable persistence.
RUN mkdir -p /data/config /data/share /data/cache /data/state /data/projects && \
    chown -R openscience:openscience /data

# Switch to non-root user
USER openscience

# Redirect XDG dirs to /data for portable persistence (survives docker compose down)
# OpenScience reads these via src/global/index.ts → XDG resolution
ENV XDG_CONFIG_HOME=/data/config
ENV XDG_DATA_HOME=/data/share
ENV XDG_CACHE_HOME=/data/cache
ENV XDG_STATE_HOME=/data/state

# Disable the browser auto-open (no GUI in container)
ENV OPENSCIENCE_NO_OPEN=1

# Default port (server binds 127.0.0.1, exposed via network_mode: host)
# Port 0 = auto-select; we pin 4096 (the server.ts default fallback)
ENV OPENSCIENCE_PORT=4096

# Healthcheck: hit the server root
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${OPENSCIENCE_PORT}/ || exit 1

EXPOSE 4096

# Run the workspace server, no browser auto-open
# --port is the only network knob (server always binds 127.0.0.1)
ENTRYPOINT ["openscience", "web", "--port", "4096"]