for i in *.wav.txt
do
echo $i
filenam="${i%.*}"
out=$(tail -n 3 $i | head -n 1 | cut -d ":" -f 2 | sed  's/[\d128-\d255]/|/g;s/|||/ /g;s/^ *//g;s/$ *//g')
fout="${filenam} ${out}"
echo $fout
echo $fout >> output.txt
done 