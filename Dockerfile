# 使用官方 Python 镜像作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 复制项目文件
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen --no-install-project

# 复制源代码和配置
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY Makefile README.md ./

# 设置环境变量
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app

# 默认运行训练脚本
CMD ["uv", "run", "python", "src/train.py"]
