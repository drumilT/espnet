
import json
from collections import defaultdict
with open('data.json', 'r') as fp:
    data = json.load(fp)

from nltk.corpus import stopwords
try:
    stop_words = set(stopwords.words('english')) 
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')) 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
  
ps = PorterStemmer()


stop_words.add("AVG_WER")
stop_words.add("z_edit_dist")
#print(stop_words)
corrects = defaultdict(dict)
for speech in data:
    for word in data[speech]:
        if word not in stop_words and ps.stem(word) not in stop_words and word.split("'")[0] not in stop_words and ps.stem(word.split("'")[0]) not in stop_words:
            corrects[speech][word]=[len(data[speech][word])-1,data[speech][word][-1]]

with open('corrects.json', 'w') as fp:
    json.dump(corrects, fp,indent=4,sort_keys=True)