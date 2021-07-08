import json
import sys
from collections import defaultdict as dd
if len(sys.argv) > 2:
    ref = json.load(open(sys.argv[1],'r'))
    hyp = json.load(open(sys.argv[2],'r'))
dump = dict({})
for speech in ref.keys():
    print(speech)
    cache = set(open("first_pass/corrects/"+speech,'r').read().splitlines())
    ref_words = ref[speech].keys()
    hyp_words = hyp[speech].keys()
    improve = dd(lambda : dd(dict))
    poor =  dd(lambda : dd(dict))
    same = dd(lambda : dd(dict))
    for word in ref_words:
        #print(word)
        if word in ["z_edit_dist","AVG_WER"]:
            continue
        if word in hyp_words:
            if len(hyp[speech][word]) > len(ref[speech][word]):
                poor[word]["hyp"] = hyp[speech][word]
                poor[word]["ref"] = ref[speech][word]
                poor[word]["cache"] = True if word in cache else False
            elif len(hyp[speech][word])== len(ref[speech][word]):
                same[word]["hyp"] = hyp[speech][word]
                same[word]["ref"] = ref[speech][word]
                same[word]["cache"] = True if word in cache else False
            else:
                improve[word]["hyp"] = hyp[speech][word]
                improve[word]["ref"] = ref[speech][word]
                improve[word]["cache"] = True if word in cache else False
        else:
            improve[word]["hyp"] = [0]
            improve[word]["ref"] = ref[speech][word]
            improve[word]["cache"] = True if word in cache else False

    for word in hyp_words:
        if word in ref_words:
            continue
        else:
            poor[word]["hyp"] = hyp[speech][word]
            poor[word]["ref"] = [0]
            poor[word]["cache"] = True if word in cache else False

    cache_negative_effect =[]
    for t in [poor,improve,same]:
        for key in t.keys():
            for w in t[key]["hyp"][:-1]:
                if w in cache:
                    cache_negative_effect.append(w)
    print("Neg Cache Effect", cache_negative_effect)
    print("len of cache",len(cache))
    for t,n in zip([poor,improve,same],["poor","improve","same"]):
        lst =[]
        for w in t.keys():
            if t[w]["cache"]:
                lst.append(w)
        print("Number of",n,"Words",len(t.keys()),"|","# in cache",len(lst),"|","# not in cache",len(t.keys())-len(lst),"|", "cache words ",lst)
    
    print("WER of hyp {} WER of ref {} WER diff (hyp-ref) {}".format(hyp[speech]["AVG_WER"],ref[speech]["AVG_WER"],hyp[speech]["AVG_WER"]-ref[speech]["AVG_WER"]))
    #print(speech,cache_negative_effect,poor.keys())
    dump[speech] = {"poor":poor,"same":same,"improve":improve}


with open('data_ref2.json', 'w') as fp:
    json.dump(dump, fp,indent=4,sort_keys=True)


