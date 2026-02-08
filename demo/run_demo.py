#!/usr/bin/env python3
"""
启动搜索Demo

运行方式:
    python demo/run_demo.py

然后打开浏览器访问: http://localhost:8000
"""

import subprocess
import sys
from pathlib import Path

# Ensure project root is on sys.path so "demo.search_api" is importable
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def check_dependencies():
    try:
        import fastapi
        import uvicorn
        print("FastAPI and Uvicorn installed")
    except ImportError:
        print("Installing FastAPI and Uvicorn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "-q"])

def main():
    check_dependencies()

    print("\n" + "=" * 50)
    print("MS MARCO Passage Search Demo")
    print("=" * 50)
    print("\nStarting server...")
    print("Open browser: http://localhost:8000")
    print("\nPress Ctrl+C to stop\n")

    import uvicorn
    uvicorn.run(
        "demo.search_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
