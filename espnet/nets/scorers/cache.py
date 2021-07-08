"""Ngram lm implement."""

from abc import ABC

#import kenlm
import torch
import numpy as np

from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorer_interface import PartialScorerInterface
import re

class Cachebase(ABC):
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
        self.lm = set(open(cache_file,"r").read().splitlines())
        #print(self.chardict)
        for i in self.chardict:
            bpe = re.sub(r'[^\x00-\x7f]',r'', i)
            if bpe.lower() in self.lm:
                print(i,"in cache")

        print(cache_file,self.lm)
        self.tmpkenlmstate = None

    def init_state(self, x):
        """Initialize tmp state."""

        return None

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
        total_words = len(self.lm)
        score = np.log(1/total_words)
        scores = torch.empty_like(next_token, dtype=x.dtype, device=y.device)
        for i, j in enumerate(next_token):
            if  re.sub(r'[^\x00-\x7f]',r'', self.chardict[j]).lower() in self.lm:
                scores[i] = score 
                print("found",self.chardict[j])
            else:
                scores[i] = 0
        return scores, None


class CacheFullScorer(Cachebase, BatchScorerInterface):
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


class CachePartScorer(Cachebase, PartialScorerInterface):
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
