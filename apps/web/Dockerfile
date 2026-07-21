FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install --no-install-recommends -y cmake g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY clone ./clone
COPY third_party ./third_party
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DIRISU_BUILD_TESTS=OFF -DIRISU_BUILD_BENCHMARKS=OFF \
    && cmake --build build --target irisu_capi -j2

FROM python:3.12-slim

ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    IRISU_CLONE_LIBRARY=/app/lib/libirisu_clone.so

WORKDIR /app
COPY python ./python
COPY apps/web ./apps/web
COPY --from=builder /src/build/libirisu_clone.so ./lib/libirisu_clone.so

EXPOSE 8000
CMD ["python3", "apps/web/server.py"]
