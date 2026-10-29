FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit==1.65.0 streamlit-agraph==0.0.45 plotly==7.1.0 httpx==0.28.1 pandas==3.0.6
COPY frontend ./frontend
COPY .streamlit ./.streamlit
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
