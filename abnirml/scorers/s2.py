from typing import Dict, List, Tuple
import pickle
import datetime
from collections import Counter
import re
import numpy as np
import pandas as pd
import kenlm
from nlpre import unidecoder
from nltk.util import ngrams
from blingfire import text_to_words
import ir_datasets
from .base import NeuralScorer



class S2(NeuralScorer):
    def __init__(self, delta=0.0):
        super().__init__(None)
        self._delta = delta

    def delta(self):
        return self._delta

    def build(self):
        if self.model is None:
            self.model = S2Ranker()


class S2Ranker:
    def __init__(self):
        super().__init__()
        #with open("/net/nfs.corp/s2-research/search_eval/search_click_ranker/trained_ranker_model_minimal.pickle", "rb") as model:
        with open("/home/sean/data/s2/trained_ranker_model_minimal.pickle", "rb") as model:
            self.model = pickle.load(model)

    def __call__(self, **inputs):
        query_texts, doc_texts = inputs['query_text'], inputs['doc_text']
        features = featureize(query_texts, doc_texts)
        return self.model.predict(features)


def featureize(query_texts, doc_texts):
    feats = []
    for query, doc in zip(query_texts, doc_texts):
        candidate = {}
        candidate["paper_year"] = 2020
        candidate["n_citations"] = 0
        candidate["n_key_citations"] = 0
        candidate["paper_title_cleaned"] = fix_text(doc)
        candidate["paper_abstract_cleaned"] = candidate["paper_title_cleaned"]
        candidate["paper_venue_cleaned"] = fix_text('arxiv')
        candidate["author_name"] = []
        feats.append(make_features(query, candidate))
    return np.array(feats)


def prepare_candidate(hit: dict) -> Dict:
    """Convert a dict to a format that features.py understands"""
    candidate = {}

    source = hit["_source"]
    author_names = [author["name"] for author in source.get("authors", [])]
    year = source.get("year", None)

    return candidate


def rerank_and_dedupe_candidates(query_string: str, raw_hits: List[dict], raw_total_hits: int) -> Tuple[int, List]:
    """Method that dedupes and reranks a candidate result set"""

    deduped_candidates = dedupe_by_id(raw_hits)
    par_candidates = build_par_candidates(query_string, deduped_candidates)
    featurized_candidates = np.array(F_POOL.starmap(make_features, par_candidates))
    predictions = get_predictions(featurized_candidates)
    adjusted_scores = posthoc_score_adjust(predictions, featurized_candidates, query_string)
    scores_argsort = np.argsort(adjusted_scores)[::-1]

    reranked_candidates = []

    for _, index in enumerate(scores_argsort):
        reranked_candidates.append(deduped_candidates[index])

    # When we dedupe the candidates we want to subtract them from the total
    # hit count in the response
    adjusted_total_hits = raw_total_hits - (len(raw_hits) - len(deduped_candidates))

    return (adjusted_total_hits, reranked_candidates)


now = datetime.datetime.now()
gorc_lm = ir_datasets.util.Lazy(lambda: kenlm.Model('/home/sean/data/s2/gorc_lm.binary'))
authors_lm = ir_datasets.util.Lazy(lambda: kenlm.Model('/home/sean/data/s2/authors_lm.binary'))
venues_lm = ir_datasets.util.Lazy(lambda: kenlm.Model('/home/sean/data/s2/venues_lm.binary'))


def nanwrapper(f, x):
    """numpy freaks out if you pass an empty arrays
    to many of its functions (like min or max).
    This wrapper just returns a nan in that case.
    """
    if len(x) == 0:
        return np.nan
    else:
        return f(x)


def remove_unigrams(s, st):
    return " ".join([i for i in s.split(" ") if i not in st])


def make_feature_names_and_constraints():
    feats = ["abstract_is_available", "paper_year_is_in_query"]

    # for lightgbm, 1 means positively monotonic, -1 means negatively monotonic and 0 means non-constraint
    constraints = ["1", "1"]

    # features just for title, abstract, venue
    for field in ["title", "abstract", "venue"]:

        feats.extend(
            [
                f"{field}_frac_of_query_matched_in_text",  # total fraction of the query that was matched in text
                f"{field}_mean_of_log_probs",  # statistics of the log-probs
                f"{field}_sum_of_log_probs*match_lens",
            ]
        )

        constraints.extend(
            ["1", "-1", "-1",]
        )

    # features for author field only
    feats.extend(
        [
            "sum_matched_authors_len_divided_by_query_len",  # total amount of (fractional wrt query) matched authors
            "max_matched_authors_len_divided_by_query_len",  # largest (fractional wrt query) author match
            "author_match_distance_from_ends",  # how far the author matches are from the front/back of the author list
        ]
    )

    constraints.extend(
        ["1", "1", "-1",]
    )

    feats.extend(
        [
            "paper_oldness",
            "paper_n_citations",  # no need for log due to decision trees
            "paper_n_key_citations",
            "paper_n_citations_divided_by_oldness",
        ]
    )

    # note: DO NOT change the paper_oldness constraint to -1
    # if you do, then seminal papers will stop being on top.
    constraints.extend(["0", "1", "1", "1"])

    feats.extend(
        [
            "fraction_of_unquoted_query_matched_across_all_fields",
            "sum_log_prob_of_unquoted_unmatched_unigrams",
            "fraction_of_quoted_query_matched_across_all_fields",
            "sum_log_prob_of_quoted_unmatched_unigrams",
        ]
    )

    constraints.extend(["1", "1", "1", "1"])

    return np.array(feats), ",".join(constraints)


def make_features(query, result_paper, max_q_len=128, max_field_len=1024):
    # the language model should have the beginning and end of sentences turned off
    lm_dict = {
        "title_abstract": lambda s: gorc_lm().score(s, eos=False, bos=False),
        "author": lambda s: authors_lm().score(s, eos=False, bos=False),
        "venue": lambda s: venues_lm().score(s, eos=False, bos=False),
    }

    # apply the language model in the field as necessary
    def lm_score(s, which_lm="title"):
        if "title" in which_lm or "abstract" in which_lm:
            return lm_dict["title_abstract"](s)
        elif "venue" in which_lm:
            return lm_dict["venue"](s)
        elif "author" in which_lm:
            return lm_dict["author"](s)
        elif "max" in which_lm:
            return np.max(
                [
                    lm_dict["title_abstract"](s),
                    lm_dict["venue"](s),
                    lm_dict["author"](s),
                ]
            )

    try:
        year = int(result_paper["paper_year"])
        year = np.minimum(now.year, year)  # papers can't be from the future.
    except:
        year = np.nan

    if result_paper["author_name"] is None:
        authors = []
    else:
        authors = result_paper["author_name"]

    # fix the text and separate out quoted and unquoted
    query = str(query)
    q = fix_text(query)[:max_q_len]
    q_quoted = [i for i in extract_from_between_quotations(q) if len(i) > 0]
    q_split_on_quotes = [i.strip() for i in q.split('"') if len(i.strip()) > 0]
    q_unquoted = [
        i.strip() for i in q_split_on_quotes if i not in q_quoted and len(i.strip()) > 0
    ]

    q_unquoted_split_set = set(" ".join(q_unquoted).split())
    q_quoted_split_set = set(" ".join(q_quoted).split())
    q_split_set = q_unquoted_split_set | q_quoted_split_set
    q_split_set -= STOPWORDS

    # we will find out how much of a match we have *across* fields
    unquoted_matched_across_fields = []
    quoted_matched_across_fields = []

    # overall features for the paper and query
    q_quoted_len = np.sum([len(i) for i in q_quoted])  # total length of quoted snippets
    q_unquoted_len = np.sum(
        [len(i) for i in q_unquoted]
    )  # total length of non-quoted snippets
    q_len = q_unquoted_len + q_quoted_len

    # if there's no query left at this point, we return NaNs
    # which the model natively supports
    if q_len == 0:
        return [np.nan] * len(FEATURE_NAMES)

    # testing whether a year is somewhere in the query and making year-based features
    if re.search(
        "\d{4}", q
    ):  # if year is in query, the feature is whether the paper year appears in the query
        year_feat = str(year) in q_split_set
    else:  # if year isn't in the query, we don't care about matching
        year_feat = np.nan

    feats = [
        result_paper["paper_abstract_cleaned"] is not None
        and len(result_paper["paper_abstract_cleaned"]) > 1,
        year_feat,  # whether the year appears anywhere in the (split) query
    ]

    # if year is matched, add it to the matched_across_all_fields but remove from query
    # so it doesn't get matched in author/title/venue/abstract later
    if np.any([str(year) in i for i in q_quoted]):
        quoted_matched_across_fields.append(str(year))
    if np.any([str(year) in i for i in q_unquoted]):
        unquoted_matched_across_fields.append(str(year))

    # if year is matched, we don't need to match it again, so removing
    if year_feat is True and len(q_split_set) > 1:
        q_split_set.remove(str(year))

    # later we will filter some features based on nonsensical unigrams in the query
    # this is the log probability lower-bound for sensible unigrams
    log_prob_nonsense = lm_score("qwertyuiop", "max")

    # features title, abstract, venue
    title_and_venue_matches = set()
    title_and_abstract_matches = set()
    for field in [
        "paper_title_cleaned",
        "paper_abstract_cleaned",
        "paper_venue_cleaned",
    ]:
        if result_paper[field] is not None:
            text = result_paper[field][:max_field_len]
        else:
            text = ""
        text_len = len(text)

        # unquoted matches
        (
            unquoted_match_spans,
            unquoted_match_text,
            unquoted_longest_starting_ngram,
        ) = find_query_ngrams_in_text(q_unquoted, text, quotes=False)
        unquoted_matched_across_fields.extend(unquoted_match_text)
        unquoted_match_len = len(unquoted_match_spans)

        # quoted matches
        (
            quoted_match_spans,
            quoted_match_text,
            quoted_longest_starting_ngram,
        ) = find_query_ngrams_in_text(q_quoted, text, quotes=True)
        quoted_matched_across_fields.extend(quoted_match_text)
        quoted_match_len = len(quoted_match_text)

        # now we (a) combine the quoted and unquoted results
        match_spans = unquoted_match_spans + quoted_match_spans
        match_text = unquoted_match_text + quoted_match_text

        # and (b) take the set of the results
        # while excluding sub-ngrams if longer ngrams are found
        # e.g. if we already have 'sentiment analysis', then 'sentiment' is excluded
        match_spans_set = []
        match_text_set = []
        for t, s in sorted(zip(match_text, match_spans), key=lambda s: len(s[0]))[::-1]:
            if t not in match_text_set and ~np.any([t in i for i in match_text_set]):
                match_spans_set.append(s)
                match_text_set.append(t)

        # remove venue results if they already entirely appeared
        if "venue" in field:
            text_unigram_len = len(text.split(" "))
            match_spans_set_filtered = []
            match_text_set_filtered = []
            for sp, tx in zip(match_spans_set, match_text_set):
                tx_unigrams = set(tx.split(" "))
                # already matched all of these unigrams in title or abstract
                condition_1 = (
                    tx_unigrams.intersection(title_and_abstract_matches) == tx_unigrams
                )
                # and matched too little of the venue text
                condition_2 = len(tx_unigrams) / text_unigram_len <= 2 / 3
                if not (condition_1 and condition_2):
                    match_spans_set_filtered.append(sp)
                    match_text_set_filtered.append(tx)

            match_spans_set = match_spans_set_filtered
            match_text_set = match_text_set_filtered

        # match_text_set but unigrams
        matched_text_unigrams = set()
        for i in match_text_set:
            i_split = i.split()
            matched_text_unigrams.update(i_split)
            if "title" in field or "venue" in field:
                title_and_venue_matches.update(i_split)
            if "title" in field or "abstract" in field:
                title_and_abstract_matches.update(i_split)

        if (
            len(match_text_set) > 0 and text_len > 0
        ):  # if any matches and the text has any length
            # log probabilities of the scores
            if "venue" in field:
                lm_probs = [lm_score(match, "venue") for match in match_text_set]
            else:
                lm_probs = [lm_score(match, "max") for match in match_text_set]

            # match character lengths
            match_lens = [len(i) for i in match_text_set]

            # match word lens
            match_word_lens = [len(i.split()) for i in match_text_set]

            # we have one feature that takes into account repetition of matches
            match_text_counter = Counter(match_text)
            match_spans_len_normed = np.log1p(list(match_text_counter.values())).sum()

            # remove stopwords from unigrams
            matched_text_unigrams -= STOPWORDS

            feats.extend(
                [
                    len(q_split_set.intersection(matched_text_unigrams))
                    / np.maximum(
                        len(q_split_set), 1
                    ),  # total fraction of the query that was matched in text
                    np.nanmean(lm_probs),  # average log-prob of the matches
                    np.nansum(
                        np.array(lm_probs) * np.array(match_word_lens)
                    ),  # sum of log-prob of matches times word-lengths
                ]
            )
        else:
            # if we have no matches, then the features are deterministically 0
            feats.extend([0, 0, 0])

    # features for author field only
    # note: we aren't using citation info
    # because we don't know which author we are matching
    # in the case of multiple authors with the same name
    q_auth = fix_author_text(query)[:max_q_len]
    q_quoted_auth = extract_from_between_quotations(q_auth)
    q_split_on_quotes = [i.strip() for i in q_auth.split('"') if len(i.strip()) > 0]
    q_unquoted_auth = [i for i in q_split_on_quotes if i not in q_quoted_auth]
    # remove any unigrams that we already matched in title or venue
    # but not abstract since citations are included there
    # note: not sure if this make sense for quotes, but keeping it for those now
    q_quoted_auth = [remove_unigrams(i, title_and_venue_matches) for i in q_quoted_auth]
    q_unquoted_auth = [
        remove_unigrams(i, title_and_venue_matches) for i in q_unquoted_auth
    ]

    unquoted_match_lens = []  # normalized author matches
    quoted_match_lens = []  # quoted author matches
    match_fracs = []
    for paper_author in authors:
        len_author = len(paper_author)
        if len_author > 0:
            # higher weight for the last name
            paper_author_weights = np.ones(len_author)
            len_last_name = len(paper_author.split(" ")[-1])
            paper_author_weights[
                -len_last_name:
            ] *= 10  # last name is ten times more important to match
            paper_author_weights /= paper_author_weights.sum()

            #
            for quotes_flag, q_loop in zip(
                [False, True], [q_unquoted_auth, q_quoted_auth]
            ):
                matched_spans, match_text, _ = find_query_ngrams_in_text(
                    q_loop,
                    paper_author,
                    quotes=quotes_flag,
                    len_filter=0,
                    remove_stopwords=True,  # only removes entire matches that are stopwords. too bad for people named 'the' or 'less'
                    use_word_boundaries=False,
                )
                if len(matched_spans) > 0:
                    matched_text_joined = " ".join(match_text)
                    # edge case: single character matches are not good
                    if len(matched_text_joined) == 1:
                        matched_text_joined = ""
                    weight = np.sum(
                        [paper_author_weights[i:j].sum() for i, j in matched_spans]
                    )
                    match_frac = np.minimum((len(matched_text_joined) / q_len), 1)
                    match_fracs.append(match_frac)
                    if quotes_flag:
                        quoted_match_lens.append(match_frac * weight)
                        quoted_matched_across_fields.append(matched_text_joined)
                    else:
                        unquoted_match_lens.append(match_frac * weight)
                        unquoted_matched_across_fields.append(matched_text_joined)
                else:
                    if quotes_flag:
                        quoted_match_lens.append(0)
                    else:
                        unquoted_match_lens.append(0)

    # since we ran this separately (per author) for quoted and uquoted, we want to avoid potential double counting
    match_lens_max = np.maximum(unquoted_match_lens, quoted_match_lens)
    nonzero_inds = np.flatnonzero(match_lens_max)
    # the closest index to the ends of author lists
    if len(nonzero_inds) == 0:
        author_ind_feature = np.nan
    else:
        author_ind_feature = np.minimum(
            nonzero_inds[0], len(authors) - 1 - nonzero_inds[-1]
        )
    feats.extend(
        [
            np.nansum(match_lens_max),  # total amount of (weighted) matched authors
            nanwrapper(np.nanmax, match_lens_max),  # largest (weighted) author match
            author_ind_feature,  # penalizing matches that are far away from ends of author list
        ]
    )

    # oldness and citations
    feats.extend(
        [
            now.year - year,  # oldness (could be nan if year is missing)
            result_paper["n_citations"],  # no need for log due to decision trees
            result_paper["n_key_citations"],
            np.nan
            if np.isnan(year)
            else result_paper["n_citations"] / (now.year - year + 1),
        ]
    )

    # special features for how much of the unquoted query was matched/unmatched across all fields
    q_unquoted_split_set -= STOPWORDS
    if len(q_unquoted_split_set) > 0:
        matched_split_set = set()
        for i in unquoted_matched_across_fields:
            matched_split_set.update(i.split())
        # making sure stopwords aren't an issue
        matched_split_set -= STOPWORDS
        # fraction of the unquery matched
        numerator = len(q_unquoted_split_set.intersection(matched_split_set))
        feats.append(numerator / np.maximum(len(q_unquoted_split_set), 1))
        # the log-prob of the unmatched unquotes
        unmatched_unquoted = q_unquoted_split_set - matched_split_set
        log_probs_unmatched_unquoted = [lm_score(i, "max") for i in unmatched_unquoted]
        feats.append(
            np.nansum(
                [i for i in log_probs_unmatched_unquoted if i > log_prob_nonsense]
            )
        )
    else:
        feats.extend([np.nan, np.nan])

    # special features for how much of the quoted query was matched/unmatched across all fields
    if len(q_quoted) > 0:
        numerator = len(set(" ".join(quoted_matched_across_fields).split()))
        feats.append(numerator / len(q_quoted_split_set))
        # the log-prob of the unmatched quotes
        unmatched_quoted = set(q_quoted) - set(quoted_matched_across_fields)
        feats.append(np.nansum([lm_score(i, "max") for i in unmatched_quoted]))
    else:
        feats.extend([np.nan, np.nan])

    return feats


# get globals to use for posthoc_score_adjust
FEATURE_NAMES, FEATURE_CONSTRAINTS = make_feature_names_and_constraints()
feature_names = list(FEATURE_NAMES)
quotes_feat_ind = feature_names.index(
    "fraction_of_quoted_query_matched_across_all_fields"
)
year_match_ind = feature_names.index("paper_year_is_in_query")
author_match_ind = feature_names.index("max_matched_authors_len_divided_by_query_len")
matched_all_ind = feature_names.index(
    "fraction_of_unquoted_query_matched_across_all_fields"
)
title_match_ind = feature_names.index("title_frac_of_query_matched_in_text")
abstract_match_ind = feature_names.index("abstract_frac_of_query_matched_in_text")
venue_match_ind = feature_names.index("venue_frac_of_query_matched_in_text")


def posthoc_score_adjust(scores, X, query=None):
    if query is None:
        query_len = 100
    else:
        query_len = len(str(query).split(" "))

    # need to modify scores if there are any quote matches
    # this ensures quoted-matching results are on top
    quotes_frac_found = X[:, quotes_feat_ind]
    has_quotes_to_match = ~np.isnan(quotes_frac_found)
    scores[has_quotes_to_match] += 1000000 * quotes_frac_found[has_quotes_to_match]

    # if there is a year match, we want to boost that a lot
    year_match = np.isclose(X[:, year_match_ind], 1.0)
    scores += 100000 * year_match

    # full author matches if the query is long enough
    if query_len > 1:
        full_author_match = np.isclose(X[:, author_match_ind], 1.0)
        scores += 100000 * full_author_match

    # then those with all ngrams matched anywhere
    matched_all_flag = np.isclose(X[:, matched_all_ind], 1.0)
    scores += 1000 * matched_all_flag

    # need to heavily penalize those with 0 percent ngram match
    matched_none_flag = np.isclose(X[:, matched_all_ind], 0.0)
    scores -= 1000 * matched_none_flag

    # find the most common match appearance pattern and upweight those
    if query_len > 1:
        if '"' in query:
            qualifying_for_cutoff = (
                np.isclose(X[:, quotes_feat_ind], 1.0) & matched_all_flag
            )
        else:
            qualifying_for_cutoff = matched_all_flag
        scores_argsort = np.argsort(scores)[::-1]
        where_zeros = np.where(qualifying_for_cutoff[scores_argsort] == 0)
        if len(where_zeros[0]) > 0:
            top_cutoff = where_zeros[0][0]
            if top_cutoff > 1:
                top_inds = scores_argsort[:top_cutoff]
                pattern_of_matches = (
                    1000
                    * (
                        (X[top_inds, title_match_ind] > 0)
                        | (X[top_inds, abstract_match_ind] > 0)
                    )
                    + 100 * (X[top_inds, author_match_ind] > 0)
                    + 10 * (X[top_inds, venue_match_ind] > 0)
                    + year_match[top_inds]
                )
                most_common_pattern = Counter(pattern_of_matches).most_common()[0][0]
                # print(Counter(pattern_of_matches).most_common())
                # don't do this if title/abstract matches are the most common
                # because usually the error is usually not irrelevant matches in author/venue
                # but usually irrelevant matches in title + abstract
                if most_common_pattern != 1000:
                    scores[top_inds[pattern_of_matches == most_common_pattern]] += 1e8

    return scores



unidecode = unidecoder()
regex_translation_table = str.maketrans("", "", "^$.\+*?{}[]()|")


def extract_from_between_quotations(text):
    """Get everything that's in double quotes
    """
    results = re.findall('"([^"]*)"', text)
    return [i.strip() for i in results]


def remove_single_non_alphanumerics(text):
    """Removes any single characters that are not
    alphanumerics and not important punctuation.
    """
    text = re.sub(r"\B[^\w\"\s]\B", "", text)
    return standardize_whitespace_length(text)


def replace_special_whitespace_chars(text: str) -> str:
    """It's annoying to deal with nonbreaking whitespace chars like u'xa0'
    or other whitespace chars.  Let's replace all of them with the standard
    char before doing any other processing."""
    text = re.sub(r"\s", " ", text)
    return text


def standardize_whitespace_length(text: str) -> str:
    """Tokenization is problematic when there are extra-long whitespaces.
    Make them all a single character in length.
    Also remove any whitespaces at beginning/end of a string"""
    return re.sub(r" +", " ", text).strip()


def fix_text(s):
    """General purpose text fixing using nlpre package
    and then tokenizing with blingfire
    """
    if pd.isnull(s):
        return ""
    s = unidecode(s)
    # fix cases when quotes are repeated
    s = re.sub('"+', '"', s)
    # dashes make quote matching difficult
    s = re.sub("-", " ", s)
    s = replace_special_whitespace_chars(s)
    # tokenize
    s = text_to_words(s).lower().strip()
    # note: removing single non-alphanumerics
    # means that we will match ngrams that are
    # usually separate by e.g. commas in the text
    # this will improve # of matches but also
    # surface false positives
    return remove_single_non_alphanumerics(s)


def fix_author_text(s):
    """Author text gets special treatment.
    No de-dashing, no tokenization, and 
    replace periods by white space.
    """
    if pd.isnull(s):
        return ""
    s = unidecode(s)
    # fix cases when quotes are repeated
    s = re.sub('"+', '"', s)
    # no periods as those make author first letter matching hard
    s = re.sub(r"\.", " ", s)
    s = replace_special_whitespace_chars(s)
    s = standardize_whitespace_length(s)
    return text_to_words(s).lower().strip()


def find_query_ngrams_in_text(
    q,
    t,
    quotes=False,
    len_filter=1,
    remove_stopwords=True,
    use_word_boundaries=True,
    max_ngram_len=7,
):
    """A function to find instances of ngrams of query q
    inside text t. Finds all possible ngrams and returns their
    character-level span.
    
    Note: because of the greedy match this function can miss
    some matches when there's repetition in the query, but
    this is likely rare enough that we can ignore it
    
    Arguments:
        q {str} -- query
        t {str} -- text
    
    Returns:
        match_spans -- a list of span tuples
        match_text_tokenized -- a list of matched tokens
    """
    longest_starting_ngram = ""

    if len(q) == 0 or len(t) == 0:
        return [], [], longest_starting_ngram
    if type(q[0]) is not str or type(t) is not str:
        return [], [], longest_starting_ngram

    q = [standardize_whitespace_length(i.translate(regex_translation_table)) for i in q]
    q = [i for i in q if len(i) > 0]
    t = standardize_whitespace_length(t.translate(regex_translation_table))

    # if not between quotes, we get all ngrams
    if quotes is False:
        match_spans = []
        match_text_tokenized = []
        for q_sub in q:
            q_split = q_sub.split()
            n_grams = []
            longest_ngram = np.minimum(max_ngram_len, len(q_split))
            for i in range(int(longest_ngram), 0, -1):
                n_grams += [
                    " ".join(ngram).replace("|", "\|") for ngram in ngrams(q_split, i)
                ]
            for i in n_grams:
                if t.startswith(i) and len(i) > len(longest_starting_ngram):
                    longest_starting_ngram = i
            if use_word_boundaries:
                matches = list(
                    re.finditer("|".join(["\\b" + i + "\\b" for i in n_grams]), t)
                )
            else:
                matches = list(re.finditer("|".join(n_grams), t))
            match_spans.extend(
                [i.span() for i in matches if i.span()[1] - i.span()[0] > len_filter]
            )
            match_text_tokenized.extend(
                [i.group() for i in matches if i.span()[1] - i.span()[0] > len_filter]
            )

        # now we remove any of the results if the entire matched ngram is just a stopword
        if remove_stopwords:
            match_spans = [
                span
                for i, span in enumerate(match_spans)
                if match_text_tokenized[i] not in STOPWORDS
            ]
            match_text_tokenized = [
                text for text in match_text_tokenized if text not in STOPWORDS
            ]

    # now matches for the between-quotes texts
    # we only care about exact matches
    else:
        match_spans = []
        match_text_tokenized = []
        for q_sub in q:
            if use_word_boundaries:
                matches = list(re.finditer("\\b" + q_sub + "\\b", t))
            else:
                matches = list(re.finditer(q_sub, t))
            if t.startswith(q_sub) and len(q_sub) > len(longest_starting_ngram):
                longest_starting_ngram = q_sub
            match_spans.extend([i.span() for i in matches])
            match_text_tokenized.extend([i.group() for i in matches])

    return match_spans, match_text_tokenized, longest_starting_ngram

STOPWORDS = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ]
)
