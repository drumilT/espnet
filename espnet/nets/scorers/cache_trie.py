
from abc import ABC
import logging
#import kenlm
import torch
import numpy as np
import re
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorer_interface import PartialScorerInterface


class CacheTriebase(ABC):
    """Ngram base implemented throught ScorerInterface."""

    def __init__(self, cache_file, token_list):
        """Initialize Ngrambase.

        Args:
            ngram_model: ngram model path
            token_list: token list from dict or model.json

        """

        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        #print(self.chardict)

        self.charlen = len(self.chardict)
        self.starts = [0]*self.charlen
        self.lowerdict = []
    
        for i in range(self.charlen):
            word = self.chardict[i]
            drop_start = re.sub(r'[^\x00-\x7f]',r'', word)
            if drop_start!=word:
                self.starts[i]=1
            self.lowerdict.append(drop_start.lower())
              
        self._end = '_end_'
        def make_trie(words):
            root = dict()
            for word in words:
                current_dict = root
                for letter in word:
                    current_dict = current_dict.setdefault(letter, {})
                current_dict[self._end] = self._end
            return root
        self.cache_words= set(open(cache_file,"r").read().splitlines())
        self.lm = make_trie(self.cache_words)
        print(cache_file,self.lm)

        #self.tmpTriestate = self.lm

    def init_state(self, x):
        """Initialize tmp state."""
        initst = dict({})
        initst["trie"] = self.lm
        #initst["sc"] =0 
        initst["inter_trie_sc"] =0 
        return initst

    def in_trie(self,trie, word):
        delta = 0.2
        #score = 0
        if trie is None:
            return None, False,0
        current_dict = trie
        for letter in word:
            if letter not in current_dict:
                return None,False,0.0
            #score += delta
            current_dict = current_dict[letter]
        return current_dict, self._end in current_dict, delta*len(word)

    def score_partial_(self, y, next_token, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        #print(x.shape)  
        ys = self.chardict[y[-1]] if y.shape[0] > 1 else None
        logging.info("Previous Token "+str(ys))
        if ys is not None:
            state = state[y[-1]]
        else:
            state = self.init_state(" ")  
        logging.info(str(state))
        #print(len(state),"Length")
        #score_o = state["sc"]
        word_end_score = 2
        scores = torch.zeros_like(next_token, dtype=x.dtype, device=y.device)
        out_state = []
        found = []
        for i, j in enumerate(next_token):
            trie_dict = state["trie"]
            inter_trie_sc = state["inter_trie_sc"]
            if self.starts[j]==1:
                trie_dict = self.lm
                inter_trie_sc = 0 
            newState,cond,score2 = self.in_trie(trie_dict ,self.lowerdict[j])
            
            if newState is None or self.starts[j]==1:
                scores[i]-= state["inter_trie_sc"]
            if cond:
                if inter_trie_sc!=0:
                    scores[i] += score2
                scores[i] += word_end_score
                inter_trie_sc = 0
                if len(found)< 10: 
                    found.append(self.chardict[j])                
            else:
                scores[i] +=score2
                inter_trie_sc += score2
            
            out_state.append({"trie":newState, 'sc':scores[i], "inter_trie_sc":inter_trie_sc})
        #print("found pos",found)
        #print("Out Side",out_state)
        #print(out_state[0])
        #print("len of outstate",len(out_state))
        return scores, out_state


class CacheTrieFullScorer(CacheTriebase, BatchScorerInterface):
    """Fullscorer for ngram."""

    def score(self, y, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            state: previous state
            x: encoded feature

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        return self.score_partial_(y, torch.tensor(range(self.charlen)), state, x)


class CacheTriePartScorer(CacheTriebase, PartialScorerInterface):
    """Partialscorer for ngram."""

    def score_partial(self, y, next_token, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        return self.score_partial_(y, next_token, state, x)

    def select_state(self, state, i):
        """Empty select state for scorer interface."""
        return state
