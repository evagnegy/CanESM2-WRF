#! /bin/bash -l

# ---- jobgen -- start
# ---- jobgen -- timestamp (2022-05-26 05:13:41)
# ---- jobgen -- generator (<class 'jobgen.generators.slurm.GPSCSlurmGenerator'>)
# ---- jobgen -- syshooks (check_memory_tmpfs,compact) (None) (None)
# ---- jobgen -- hooks (None) (None) (None)
#
#SBATCH --job-name=submit_interp.job
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/submit_interp_eccc_pr_rcp45_3.out
#SBATCH --partition=standard
#SBATCH --account=dfo_pfm
#SBATCH --no-requeue
#SBATCH --export=USER,LOGNAME,HOME,MAIL,PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin,JOBGEN_JOINOUTERR=true,JOBGEN_NAME=submit_interp.jgen,JOBGEN_NSLOTS=1,JOBGEN_OUTPATH=/gpfs/fs7/dfo/hpcmc/pfm/evg000/verification_scripts/submit_interp.out,JOBGEN_PROJECT=dfo_pfm,JOBGEN_QUEUE=dev,JOBGEN_SHELL=/bin/bash,JOBGEN_SLOT_IMAGE=dfo/dfo_all_default_ubuntu-18.04-amd64_latest,JOBGEN_SLOT_MEMORY=250000M,JOBGEN_SLOT_NCORES=64,JOBGEN_SLOT_TMPFS=1000M,JOBGEN_WALLCLOCK=6:00:00
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

SRCDIR=~/space/CanESM2-WRF/scripts/
cd $SRCDIR


WRFPATH=~/evg000/gpfs7/rcp45_verification/
station_list=/gpfs/fs7/dfo/hpcmc/pfm/spfm000/CanESM2-WRF/obs/NOAA_d03_stations.csv
#station_list=~/gpfs7/obs/BCH_d03_stations.csv

time ./get_wrf_interpolation_rcp.sh pr 3 $WRFPATH $station_list rcp45
#time ./get_wrf_interpolation_BCH.sh wind 2 $WRFPATH $station_list
#time ./get_canesm2raw_interpolation.sh $WRFPATH $station_list

#time ./get_wrf_interpolation.sh wind 3 $WRFPATH $station_list
#time ./get_wrf_interpolation.sh wdir 3 $WRFPATH $station_list
#time ./get_wrf_interpolation.sh wind 2 $WRFPATH $station_list
#time ./get_wrf_interpolation.sh wdir 2 $WRFPATH $station_list
#time ./get_wrf_interpolation.sh wind 1 $WRFPATH $station_list
#time ./get_wrf_interpolation.sh wdir 1 $WRFPATH $station_list

