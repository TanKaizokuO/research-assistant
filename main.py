from numpy import rint

from logger import configure_logging
import logging
from data_loaders.web_search import Tavily_Client
from dotenv import load_dotenv
import os

load_dotenv()


if __name__ == "__main__":
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    configure_logging()
    sources = Tavily_Client(tavily_api_key)
    logging.info(f"Retrieved {len(sources)} sources.")
