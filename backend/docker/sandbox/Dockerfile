# Sandboxed code-execution image (app/tasks/sandbox.py). Built once, locally,
# with network access (the same "download once, then air-gapped forever"
# pattern as `ollama pull` or the Phase 3 reranker's HF download) — every
# actual run of this image afterward is `docker run --network none`, so
# nothing it does at execution time can reach the network, including a
# `pip install` a generated script might try.
#
# numpy/scipy are pre-installed because they're the most common thing a
# real analysis/statistics request needs, and the coding model has no way
# to install anything itself once the container is running with no network
# — verified live: without this, a stats-heavy request (regression with a
# confidence interval) burned several minutes retrying numpy/scipy/
# statsmodels imports that could never succeed, one of the real
# contributors to a "stuck" Console alongside the (separately fixed)
# missing-thinking-stream issue.
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy scipy
