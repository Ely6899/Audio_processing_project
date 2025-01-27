import sys
from pathlib import Path

import numpy as np

from GeneralUtils.pipeline_utils import encoder_inference

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python script1.py <file_path>")
        sys.exit(1)

    file_path: Path = Path(sys.argv[1])
    embedding = encoder_inference(file_path)
    np.save(sys.stdout.buffer, embedding)


