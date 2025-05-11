#!/bin/bash

# Initialize conda for shell usage
echo "🔧 Running conda init..."
conda init

# Refresh shell session
echo "🔁 Reloading shell..."
source ~/.bashrc

# Create RAPIDS conda environment
echo "🚀 Creating RAPIDS conda environment..."
conda create -y -n rapids -c rapidsai -c nvidia -c conda-forge \
    cuml=23.02 python=3.10 cudatoolkit=11.8

# Activate the new environment
echo "✅ Activating RAPIDS environment..."
source ~/.bashrc
conda activate rapids

# Install Jupyter kernel for this environment
echo "🧠 Registering new Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name rapids --display-name "Python (RAPIDS)"

echo "🎉 Done! Now refresh your Jupyter tab and select 'Python (RAPIDS)' kernel!"
