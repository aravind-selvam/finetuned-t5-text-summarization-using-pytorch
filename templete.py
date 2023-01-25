# Creating empty files in the directory structure.
import os
from pathlib import Path

list_of_files = [
    f"ts/__init__.py",
    f"ts/cloud_storage/__init__.py",
    f"ts/components/__init__.py",
    f"ts/constants/__init__.py",
    f"ts/entity/__init__.py",
    f"ts/exceptions/__init__.py",
    f"ts/logger/__init__.py",
    f"ts/pipeline/__init__.py",
    f"ts/utils/__init__.py",
    f'setup.py'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")