# main.py — entry point for the Research Assistant FastAPI backend.
# Fixes two bugs from the original:
#   1. `from data_loaders.web_search import Tavily_Client` — that export does
#      not exist; the correct callable is `manual_web_search`.
#   2. `from numpy import rint` — unused import, not needed here.
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
