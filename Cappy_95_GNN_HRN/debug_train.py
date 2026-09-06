#!/usr/bin/env python3
"""Debug training script."""

import sys
import os
os.chdir('d:\\capstone\\GRITSENSE\\Cappy_95_GNN_HRN')
sys.path.insert(0, '.')

try:
    print("1. Importing modules...")
    from training.train_with_real_data import main
    print("2. Import successful. Starting main()...")
    main()
    print("3. Main completed.")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
