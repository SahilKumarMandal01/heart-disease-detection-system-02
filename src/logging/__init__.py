import logging
import os
from datetime import datetime

# Generate timestamped log filename
LOG_FILENAME = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"

# Prepare log directory path
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] [line:%(lineno)d] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
