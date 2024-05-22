#!/bin/bash
#SBATCH --job-name=train_model_improve
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --ntasks=25
#SBATCH --time=24:00:00
#SBATCH --mem=700G

# Record start time
start_time=$(date +%s.%N)

# Run Python script
conda run -n bdao python /group/sbs007/bdao/project/scripts/cpu/train_model_improve.py

# Record end time
end_time=$(date +%s.%N)

# Calculate real-time duration
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Real-time duration: $execution_time seconds"



