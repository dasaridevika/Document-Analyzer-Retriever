import os
import sys
import time
import subprocess

def main():
    port = os.getenv("PORT", "8501")
    backend_port = os.getenv("BACKEND_PORT", "8001")

    print(f"=========================================================")
    print(f" Starting Document Analyser & Retriever Single-Service")
    print(f" FastAPI Backend: http://127.0.0.1:{backend_port}")
    print(f" Streamlit Frontend: http://0.0.0.0:{port}")
    print(f"=========================================================")

    # 1. Launch FastAPI Backend process
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", str(backend_port)
    ]
    print(f"Launching FastAPI backend on port {backend_port}...")
    backend_process = subprocess.Popen(backend_cmd)

    # Brief delay to let backend initialize
    time.sleep(3)

    # 2. Launch Streamlit Frontend process
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    print(f"Launching Streamlit frontend on port {port}...")
    frontend_process = subprocess.Popen(frontend_cmd)

    # Monitor processes
    try:
        while True:
            b_code = backend_process.poll()
            f_code = frontend_process.poll()

            if b_code is not None:
                print(f"ERROR: FastAPI backend exited unexpectedly with code {b_code}")
                frontend_process.terminate()
                sys.exit(b_code)

            if f_code is not None:
                print(f"ERROR: Streamlit frontend exited unexpectedly with code {f_code}")
                backend_process.terminate()
                sys.exit(f_code)

            time.sleep(2)

    except KeyboardInterrupt:
        print("Shutting down processes...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
