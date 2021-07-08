# for i in TEDLIUM/TEDLIUM_release1/test/seg/*; do
#    j=$(basename $i)

#    if [[ ! -f test2/$j.txt ]]; then
#        echo $i
#        CUDA_VISIBLE_DEVICES=3 bash ../../../utils/recog_wav.sh --models librispeech.transformer.v1 $i >test2/$j.txt
#    fi
# done

# for i in  DanielKahneman_2010-0096588-0097057.wav  DanielKahneman_2010-0097057-0098070.wav  EricMead_2009P-0001843-0003166.wav  GaryFlake_2010-0001605-0002712.wav  JamesCameron_2010-0001675-0002796.wav  JaneMcGonigal_2010-0006200-0006901.wav  JaneMcGonigal_2010-0051044-0051590.wav  JaneMcGonigal_2010-0051590-0052984.wav  JaneMcGonigal_2010-0060796-0062073.wav  JaneMcGonigal_2010-0070076-0070540.wav  JaneMcGonigal_2010-0070540-0071959.wav  JaneMcGonigal_2010-0071959-0072539.wav  JaneMcGonigal_2010-0080708-0081759.wav  JaneMcGonigal_2010-0081759-0082734.wav  JaneMcGonigal_2010-0090306-0091140.wav  JaneMcGonigal_2010-0091140-0091922.wav  JaneMcGonigal_2010-0091922-0092673.wav 
# do
#     echo $i
#     CUDA_VISIBLE_DEVICES=3 bash ../../../utils/recog_wav.sh --models librispeech.transformer.v1 TEDLIUM/TEDLIUM_release1/test/seg/$i >test2/$i.txt
# done

i=$1
j=$(basename $i)
if [[ ! -f test_lm/$j.txt ]]; then
echo $i 
bash ../../../utils/recog_wav.sh --models librispeech.transformer.v1 $i >test_lm/$j.txt 
fi