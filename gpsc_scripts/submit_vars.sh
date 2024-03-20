
var=wind
domain=d03

cd /home/spfm000/space/CanESM2-WRF/scripts/

cp run_vars.sh run_vars_1.sh
cp run_vars.sh run_vars_2.sh
cp run_vars.sh run_vars_3.sh

sed -i s%M_START%1%g ./run_vars_1.sh
sed -i s%M_END%4%g ./run_vars_1.sh
sed -i s%M_START%5%g ./run_vars_2.sh
sed -i s%M_END%8%g ./run_vars_2.sh
sed -i s%M_START%9%g ./run_vars_3.sh
sed -i s%M_END%12%g ./run_vars_3.sh

sed -i s%VAR%${var}%g ./run_vars_1.sh
sed -i s%VAR%${var}%g ./run_vars_2.sh
sed -i s%VAR%${var}%g ./run_vars_3.sh
sed -i s%DOMAIN%${domain}%g ./run_vars_1.sh
sed -i s%DOMAIN%${domain}%g ./run_vars_2.sh
sed -i s%DOMAIN%${domain}%g ./run_vars_3.sh

year=$1

sed -i s%YEAR%${year}%g ./run_vars_1.sh
sed -i s%YEAR%${year}%g ./run_vars_2.sh
sed -i s%YEAR%${year}%g ./run_vars_3.sh

sbatch ./run_vars_1.sh 
sbatch ./run_vars_2.sh
sbatch ./run_vars_3.sh


