@echo off
echo ===================================================
echo Starting DevLens AI Backend (FastAPI + Uvicorn)
echo Using Virtual Environment Python: backend\.venv
echo ===================================================
cd backend
.venv\Scripts\python.exe app\main.py
pause

