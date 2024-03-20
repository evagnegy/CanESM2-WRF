#! /bin/bash -l

# ---- jobgen -- start
# ---- jobgen -- timestamp (2022-05-26 05:13:41)
# ---- jobgen -- generator (<class 'jobgen.generators.slurm.GPSCSlurmGenerator'>)
# ---- jobgen -- syshooks (check_memory_tmpfs,compact) (None) (None)
# ---- jobgen -- hooks (None) (None) (None)
#
#SBATCH --job-name=submit_vars.job
#SBATCH --open-mode=append
#SBATCH --output=/home/spfm000/space/CanESM2-WRF/submit_vars.out
#SBATCH --partition=standard
#SBATCH --account=dfo_pfm
#SBATCH --no-requeue
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin,JOBGEN_JOINOUTERR=true,JOBGEN_NAME=submit_vars.jgen,JOBGEN_NSLOTS=1,JOBGEN_OUTPATH=/gpfs/fs7/dfo/hpcmc/pfm/evg000/verification_scripts/submit_vars.out,JOBGEN_PROJECT=dfo_pfm,JOBGEN_QUEUE=dev,JOBGEN_SHELL=/bin/bash,JOBGEN_SLOT_IMAGE=dfo/dfo_all_default_ubuntu-18.04-amd64_latest,JOBGEN_SLOT_MEMORY=250000M,JOBGEN_SLOT_NCORES=64,JOBGEN_SLOT_TMPFS=1000M,JOBGEN_WALLCLOCK=6:00:00
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=250000M
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

SRCDIR=/home/spfm000/space/CanESM2-WRF/scripts/
cd $SRCDIR

run="rcp45"

WRFDIR=/gpfs/fs7/dfo/hpcmc/pfm/evg000/CanESM2_WRF_runs/${run}_r1i1p1_2050/WRF/
OUTDIR=/home/spfm000/space/CanESM2-WRF/${run}/variables_complete

time python getwind.py wind $WRFDIR $OUTDIR 2050 5 8 d03


