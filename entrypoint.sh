#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI server..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait for FastAPI to be up
sleep 3

# Disable Streamlit telemetry and headless mode to bypass email prompt
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true

# Start Streamlit frontend in the foreground
echo "Starting Streamlit dashboard..."
python -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Trap SIGINT and SIGTERM to kill both processes
trap "kill $FASTAPI_PID $STREAMLIT_PID" SIGINT SIGTERM

# Wait for any process to exit
wait -n

exit $?
