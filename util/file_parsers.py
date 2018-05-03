"""
This module handles parsing of various file formats
"""

import re


_allowable_types = ["tweets", "lyrics", "tweets2"]

def parse(filename, filetype=None, bow_file=None, artist_name=None):
    """
    filename: the name of the file to open and parse
    filetype - optional: The type of the file, one of [_allowable_types]
        If no filetype is specified, it will be treated as plain text
    bow_file - optional: the name of the bag of words file when parsing the mxm dataset
    artist_name - optional: the desired artist when parsing the mxm dataset
    """
    if filetype is None:
        return parse_raw(filename)
    elif filetype not in _allowable_types:
        print("Unsupported file type %s specified. Allowed types are %s".format(type, _allowable_types))
    elif filetype == "tweets":
        return parse_tweets(filename)
    elif filetype == "tweets2":
        return parse_tweets2(filename)
    elif filetype == "lyrics":
        if bow_file is None or artist_name is None:
            print("Bag of word file or artist name not specified but lyric parsing requested")
        else:
            return parse_mxm(filename, bow_file, artist_name)

# For parsing this csv format
# https://www.kaggle.com/austinvernsonger/donaldtrumptweets
def parse_tweets(filename):
    f = open(filename)
    tweet_texts = []
    for line in f:
        tweet_data = line.split(',')
        if len(tweet_data) > 3:
            tweet_text = tweet_data[2]
            # ignore retweets
            if tweet_text.startswith("RT"):
                continue
            tweet_texts.append(tweet_text)
            # TODO: remove urls
    return tweet_texts

# For parsing this csv format
# https://www.kaggle.com/speckledpingu/RawTwitterFeeds/
def parse_tweets2(filename):
    f = open(filename)
    tweet_texts = []
    for line in f:
        tweet_data = line.split(',')
        # if not a retweet
        if len(tweet_data) > 5 and tweet_data[4] == "False":
            tweet_text = tweet_data[5]
            tweet_text.replace('.', '')
            tweet_text = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet_text)
            tweet_text = re.sub(r'pic.twitter.com.*[\r\n]*', '', tweet_text)

            tweet_texts.append(tweet_text)
    return tweet_texts


# For parsing text line by line. Assumes empty line in a doc separator unless otherwise specified
def parse_raw(filename, doc_separator='\n'):
    f = open(filename)
    docs = []
    doc = ''
    for l in f:
        if l == doc_separator:
            docs.append(doc)
            doc = ''
        else:
            docs.append(l)
    return docs

# For parsing mxm data https://labrosa.ee.columbia.edu/millionsong/musixmatch
def parse_mxm(matches_file, bow_file, requested_artist):
    matches = open(matches_file)  # "mxm_779k_matches.txt"
    artist_tracks = []

    for l in matches:
        if l.startswith("#"):
            continue
        else:
            items = l.split("<SEP>")
            print (items)
            artist_name = items[1]
            if artist_name.lower().strip() == requested_artist.lower().strip():
               artist_tracks.append(items[0])
    print("Artist has " + str(len(artist_tracks)) + " tracks")

    f = open(bow_file)
    song_bows = []
    for l in f:
        top_words = []
        if l.startswith("#"):
            continue
        elif l.startswith("%"):
            for word in l[1:].split(","):
                top_words.append(word)
        else:
            items = l[1:].split(",")
            msd_track_id = items[0]
            if msd_track_id in artist_tracks:
                bow = {}
                for word_count in items[2:]:
                    (index, count) = word_count.split(":")
                    bow[top_words[index]] = count
                song_bows.append(bow)
