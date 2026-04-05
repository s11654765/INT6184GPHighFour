# 构建：在 obesity_web 目录执行
#   docker build -t obesity-predict .
# 运行：docker run -p 8080:8080 -e PORT=8080 obesity-predict
# 部署到 Render / Railway / Fly.io 时，平台会注入 PORT，无需学员安装任何环境。

FROM python:3.11-slim

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY app.py feature_mappings.py model_final_encode.py student_advice_en.py ./
COPY static ./static/
COPY xgb_model_bundle.joblib ./

ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 app:app
