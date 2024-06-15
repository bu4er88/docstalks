import os
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

CLIENT_ID = os.environ.get('client-id', None)
CLIENT_SECRET = os.environ.get('client-secret', None)