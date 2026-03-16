from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bookcast.demo_tts import main


if __name__ == "__main__":
    main()
