import json 
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

perc = 0.1
with open('corrects.json', 'r') as fp:
    corrects = json.load(fp)

tfidf = TfidfVectorizer(analyzer='word', stop_words = 'english')
speech = defaultdict(str)

for line in open("../first_pass/text").read().splitlines():
    speech[line.split("-")[0]]+=(" ".join(line.split()[1:])+" ")

X_train = list(speech.values())
# fit_transform on training data
X_tfidf = tfidf.fit_transform(X_train)
fet_names=defaultdict(int)
for i,j in enumerate(tfidf.get_feature_names()):
    fet_names[j]=i
#print(fet_name)


select = dict({})
for speaker in corrects.keys():
    words = list(corrects[speaker].keys())
    num_mistakes = sum([corrects[speaker][word][1] for word in words])
    Y = tfidf.transform([speech[speaker]])[0].toarray()[0]
    words.sort(key=lambda x: Y[fet_names[x]],reverse=True)
    select[speaker] = words[:int(num_mistakes*perc)]


#implement tf-id based choosing
#prefix of the talk (10%)


with open(str(perc)+'_corrects_to_add.json', 'w') as fp:
    json.dump(select, fp,indent=4,sort_keys=True)

for speaker in corrects.keys():
    with open("corrects/"+speaker,"w") as fp:
        for w in select[speaker]:
            fp.write(" ".join([w])+"\n")