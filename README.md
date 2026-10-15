e host system uses an externally‑managed Python installation, so installing packages globally is blocked. Let’s create an isolated virtual environment:

bash

# From the project root

cd /home/ansh/projects/hackathon/PayProtect/PayGuard00

# 1️⃣ Create a virtualenv called .venv

python3 -m venv .venv

# 2️⃣ Activate it (you’ll see the prompt prefixed with (.venv))

source .venv/bin/activate

# 3️⃣ Upgrade pip (optional but helps)

pip install --upgrade pip
Why?
All required libraries (pandas, numpy, scikit‑learn, joblib, matplotlib, seaborn, networkx) will be installed inside this environment, leaving the system Python untouched.

2️⃣ Install the project’s Python dependencies
bash

pip install -r requirements.txt
What you’ll see – a list of packages being downloaded and compiled. If you encounter any compilation errors (e.g., missing gcc), the easiest workaround on a typical Debian‑based sandbox is to use the pre‑compiled wheels that pip fetches automatically.

3️⃣ Run the whole pipeline (one‑liner)
A tiny helper script called run_all.sh is provided for you. If you prefer to run each step manually, skip to the manual section below.

bash

# Make it executable (only needed once)

chmod +x run_all.sh

# Execute the pipeline

./run_all.sh
What run_all.sh does (for transparency):

bash

#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Generate data

python3 ml/generate_dataset.py

# 2️⃣ Build features & split

python3 ml/build_features.py

# 3️⃣ Train the three models

python3 ml/train_supervised.py
python3 ml/train_anomaly.py
python3 ml/train_graph.py

# 4️⃣ Evaluate & create plots + markdown report

python3 ml/evaluate.py
echo "✅ All steps finished! Check the 'reports' directory."
Running the script will print a short progress line after each step, e.g.:

Synthetic dataset generated at data/synthetic_transactions.csv
Features built, splits saved under data/
Supervised RandomForest trained – model saved at ml/models/rf.joblib
Anomaly IsolationForest trained – model saved at ml/models/anomaly.joblib
Graph model trained – model saved at ml/models/graph_lr.joblib
Evaluation report written to reports/evaluation_report.md
4️⃣ Manually running each stage (optional)
If you’d rather invoke the scripts one at a time (useful for debugging), here are the exact commands in the correct order:

bash

# 4.1 Generate synthetic data

python3 ml/generate_dataset.py

# 4.2 Build features & split (creates:

# data/X_train.npy, data/y_train.npy, data/X_val.npy, data/y_val.npy,

# data/X_test.npy, data/y_test.npy, plus a pickled encoder)

python3 ml/build_features.py

# 4.3 Train the supervised RandomForest

python3 ml/train_supervised.py

# 4.4 Train the anomaly IsolationForest

python3 ml/train_anomaly.py

# 4.5 Train the graph‑based LogisticRegression

python3 ml/train_graph.py

# 4.6 Evaluate everything & plot

python3 ml/evaluate.py
