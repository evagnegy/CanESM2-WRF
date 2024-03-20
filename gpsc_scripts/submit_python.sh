#! /bin/bash -l

# ---- jobgen -- start
# ---- jobgen -- timestamp (2022-05-26 05:13:41)
# ---- jobgen -- generator (<class 'jobgen.generators.slurm.GPSCSlurmGenerator'>)
# ---- jobgen -- syshooks (check_memory_tmpfs,compact) (None) (None)
# ---- jobgen -- hooks (None) (None) (None)
#
#SBATCH --job-name=submit_pr.job
#SBATCH --open-mode=append
#SBATCH --output=/home/spfm000/space/CanESM2-WRF/scripts/python.out
#SBATCH --partition=standard
#SBATCH --account=dfo_pfm
#SBATCH --no-requeue
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=450000M
#SBATCH --comment="image=dfo/dfo_all_default_ubuntu-18.04-amd64_latest,ssh=true,nsswitch=true,tmpfs_size=1000M"
#
# ---- jobgen -- end

export SLURM_EXPORT_ENV=ALL


echo '#################### GPSC CONFIG ####################'

export SSM_VERBOSE=1
export SSM_DEBUG=1

ulimit -s unlimited
ulimit -l unlimited
ulimit -n 4096

# Use intel oneapi compilers
. ssmuse-sh -x main/opt/intelcomp/inteloneapi-2021.3.0
source ~/.profile

echo '*******************'
echo '* RUN * RUN * RUN *'
echo '*******************'

#conda deactivate
#cd ~/space/envs/
#conda activate xclim_env/

SRCDIR=~/space/CanESM2-WRF/scripts/
cd $SRCDIR

time python calc_climdex.py


