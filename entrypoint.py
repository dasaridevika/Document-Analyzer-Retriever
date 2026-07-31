import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

def wait_for_backend(url: str, timeout: int = 30) -> bool:
    """
    Polls the backend health endpoint until it responds with HTTP 200 OK.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False

def main():
    port = os.getenv("PORT", "8501")
    backend_port = os.getenv("BACKEND_PORT", "8001")

    print(f"=========================================================")
    print(f" Starting Document Analyser & Retriever Single-Service")
    print(f" FastAPI Backend Port: {backend_port}")
    print(f" Streamlit Frontend Port: {port}")
    print(f"=========================================================")

    # 1. Launch FastAPI Backend process (binding to 0.0.0.0:8001)
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0",
        "--port", str(backend_port)
    ]
    print(f"Launching FastAPI backend on 0.0.0.0:{backend_port}...")
    backend_process = subprocess.Popen(backend_cmd)

    # 2. Wait for FastAPI backend to become healthy
    health_url = f"http://127.0.0.1:{backend_port}/api/health"
    print(f"Waiting for FastAPI backend to respond at {health_url}...")
    
    healthy = wait_for_backend(health_url, timeout=35)
    if not healthy:
        print(f"CRITICAL ERROR: FastAPI backend failed to start on port {backend_port} within 35 seconds!")
        backend_process.terminate()
        sys.exit(1)

    print(f"✅ FastAPI backend is HEALTHY and ready!")

    # 3. Launch Streamlit Frontend process
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
                if frontend_process.poll() is None:
                    frontend_process.terminate()
                sys.exit(b_code)

            if f_code is not None:
                print(f"ERROR: Streamlit frontend exited unexpectedly with code {f_code}")
                if backend_process.poll() is None:
                    backend_process.terminate()
                sys.exit(f_code)

            time.sleep(2)

    except KeyboardInterrupt:
        print("Shutting down processes...")
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    main()
