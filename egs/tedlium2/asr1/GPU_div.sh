n=$1
ls TEDLIUM/TEDLIUM_release1/test/seg/ > list_temp
split -n l/$n list_temp div_rem
#rm list_temp
count=1
for i in div_rem*
do 
    $(cat $i | xargs -P 8 -I {} bash check_test3.sh TEDLIUM/TEDLIUM_release1/test/seg/{} $count)  &
    count=$((count+1))
done 
#rm div_rem*
