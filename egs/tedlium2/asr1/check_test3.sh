i=$1
j=$(basename $i)
count=$2
if [[ ! -f test_with_cache_acoustic_tfidf3/$j.txt ]]; then
echo $i 
export CUDA_VISIBLE_DEVICES=$count
bash ../../../utils/recog_wav3.sh --models librispeech.transformer.v1 $i > test_with_cache_acoustic_tfidf3/$j.txt 
fi
