FROM python:3.12-slim

WORKDIR /app

# システム依存（portaudio は sounddevice に必要だが Cloud Run では縮退動作）
# pip install uv を同一レイヤーにまとめてレイヤー数削減
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# 依存関係をコピーしてインストール
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 非 root ユーザー作成（uv キャッシュ用にホームディレクトリが必要）
RUN adduser --disabled-password appuser

# アプリケーションコードをコピー（非 root 所有）
COPY --chown=appuser:appuser sidecar/ sidecar/
COPY --chown=appuser:appuser data/ data/

# .venv の所有権を appuser に変更（uv run で書き込みが必要）
RUN chown -R appuser:appuser /app/.venv

USER appuser

ENV PORT=8765
ENV HOST=0.0.0.0

EXPOSE 8765

CMD ["uv", "run", "python", "-m", "sidecar.main"]
