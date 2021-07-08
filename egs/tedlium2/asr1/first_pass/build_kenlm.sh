
for i in corrects/*
do 
    j=$(basename $i)
    mkdir -p kenlm_models/$j
    ../kenlm_build/bin/lmplz -o 1 --verbose_header --text $i --arpa kenlm_models/$j/log.arpa --vocab_estimate 50 --discount_fallback
    #../kenlm_build/bin/build_binary -s  kenlm_models/$j/log.arpa  kenlm_models/$j/log.bin
done