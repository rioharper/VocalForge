import re
import regex
import os
from nemo_text_processing.text_normalization.normalize import Normalizer
from num2words import num2words
from .normalization_helpers import RU_ABBREVIATIONS, LATIN_TO_RU

def get_unicode(language):
    lower_case_unicode = ''
    upper_case_unicode = ''
    if language == "ru":
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'
    elif language not in ["ru", "en"]:
        print(f"Consider using {language} unicode letters for better sentence split.")
    return lower_case_unicode, upper_case_unicode


def get_vocabulary_symbols(vocabulary):
    vocabulary_symbols = []
    for x in vocabulary:
        if x != "<unk>":
            # for BPE models
            vocabulary_symbols.extend([x for x in x.replace("##", "").replace("▁", "")])
    vocabulary_symbols = list(set(vocabulary_symbols))
    vocabulary_symbols += [x.upper() for x in vocabulary_symbols]
    return vocabulary_symbols


def format_text(text_file: str, language: str) -> str:
    """
    Formats the text in the given text file to remove symbols and symbols that might cause trouble
    in sentence splitting.

    :param text_file: str: path to the text file
    :param language: str: language code for the text file. (Currently supported 'ru', 'en')
    :return: str: Formatted text string
    """
    with open(text_file, "r", encoding='utf-8') as f:
        transcript = f.read()

    print(f"Splitting text in {text_file} into sentences.")
    # remove some symbols for better split into sentences
    transcript = (
        transcript.replace("\n", " ")
        .replace("\t", " ")
        .replace("…", "...")
        .replace("\\", " ")
        .replace("--", " -- ")
        .replace(". . .", "...")
    )

    # end of quoted speech - to be able to split sentences by full stop
    transcript = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", transcript)

    # remove extra space
    transcript = re.sub(r" +", " ", transcript)

    #remove harmful characters
    transcript = re.sub(r'(\[.*?\])', ' ', transcript)
    # remove text in curly brackets
    transcript = re.sub(r'(\{.*?\})', ' ', transcript)

    lower_case_unicode = ''
    upper_case_unicode = ''
    if language == "ru":
        lower_case_unicode = '\u0430-\u04FF'
        upper_case_unicode = '\u0410-\u042F'
    elif language not in ["ru", "en"]:
        print(f"Consider using {language} unicode letters for better sentence split.")

    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(r'[a-z' + lower_case_unicode + ']\.\s[a-z' + lower_case_unicode + ']\.', transcript)
    for match in matches:
        transcript = transcript.replace(match, match.replace('. ', '.'))
    return transcript


def split_text(transcript: str, language: str, vocabulary: list, 
               max_length: int, additional_split_symbols:str =None, min_length=0) -> list:
    """
    Split the given text into smaller segments.

    Arguments:
        transcript (str): The transcript to be split.
        language (str): The language of the text.
        vocabulary (List[str]): List of vocabulary words.
        max_length (int): Maximum length of the segments.
        additional_split_symbols (Optional[str], optional): Additional symbols to be used 
            for splitting the text. Default is None.

    Returns:
        List[str]: List of smaller segments of the given text.
    """
    # find phrases in quotes
    with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', transcript)
    sentences = []
    last_idx = 0
    for m in with_quotes:
        match = m.group()
        match_idx = m.start()
        if last_idx < match_idx:
            sentences.append(transcript[last_idx:match_idx])
        sentences.append(match)
        last_idx = m.end()
    sentences.append(transcript[last_idx:])
    sentences = [s.strip() for s in sentences if s.strip()]

    # Read and split transcript by utterance (roughly, sentences)
    lower_case_unicode, upper_case_unicode = get_unicode(language)
    split_pattern = f"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]\.)"
    split_pattern += f"(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"

    new_sentences = []
    for sent in sentences:
        new_sentences.extend(regex.split(split_pattern, sent))
    sentences = [s.strip() for s in new_sentences if s.strip()]

    # Additional split on symbols
    def additional_split(sentences, split_on_symbols):
        if len(split_on_symbols) == 0:
            return sentences

        split_on_symbols = split_on_symbols.split("|")

        def _split(sentences, delimiter):
            result = []
            for sent in sentences:
                split_sent = sent.split(delimiter)
                # keep the delimiter
                split_sent = [(s + delimiter).strip() for s in split_sent[:-1]] + [split_sent[-1]]

                if "," in delimiter:
                    # split based on comma usually results in too short utterance, combine sentences
                    # that result in a single word split. It's usually not recommended to do that for other delimiters.
                    comb = []
                    for s in split_sent:
                        MIN_LEN = 2
                        # if the previous sentence is too short, combine it with the current sentence
                        if len(comb) > 0 and (len(comb[-1].split()) <= MIN_LEN or len(s.split()) <= MIN_LEN):
                            comb[-1] = comb[-1] + " " + s
                        else:
                            comb.append(s)
                    result.extend(comb)
                else:
                    result.extend(split_sent)
            return result.rstrip()

        another_sent_split = []
        for sent in sentences:
            split_sent = [sent]
            for delimiter in split_on_symbols:
                if len(delimiter) == 0:
                    continue
                split_sent = _split(split_sent, delimiter + " " if delimiter != " " else delimiter)
            another_sent_split.extend(split_sent)

        sentences = [s.strip() for s in another_sent_split if s.strip()]
        return sentences
    if additional_split_symbols != None:
        additional_split_symbols = additional_split_symbols.replace("/s", " ")
        sentences = additional_split(sentences, additional_split_symbols)

    vocabulary_symbols = []
    for x in vocabulary:
        if x != "<unk>":
            # for BPE models
            vocabulary_symbols.extend([x for x in x.replace("##", "").replace("▁", "")])
    vocabulary_symbols = list(set(vocabulary_symbols))
    vocabulary_symbols += [x.upper() for x in vocabulary_symbols]

    # check to make sure there will be no utterances for segmentation with only OOV symbols
    vocab_no_space_with_digits = set(vocabulary_symbols + [str(i) for i in range(10)])
    if " " in vocab_no_space_with_digits:
        vocab_no_space_with_digits.remove(" ")

    sentences = [
        s.strip() for s in sentences if len(vocab_no_space_with_digits.intersection(set(s.lower()))) > 0 and s.strip()
    ]

    # when no punctuation marks present in the input text, split based on max_length
    if len(sentences) == 1:
        sent = sentences[0].split()
        sentences = []
        for i in range(0, len(sent), max_length):
            sentences.append("".join(sent[i: i + max_length]))
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= min_length]
    return sentences

def normalize_text(sentences, language, vocabulary, n_jobs=1, batch_size=100):
    # substitute common abbreviations before applying lower case
    def substitute(sentences):
        if language == "ru":
            for k, v in RU_ABBREVIATIONS.items():
                sentences = [s.replace(k, v) for s in sentences]
            # replace Latin characters with Russian
            for k, v in LATIN_TO_RU.items():
                sentences = [s.replace(k, v) for s in sentences]

        if language == "en":
            print("Using NeMo normalization tool...")
            normalizer = Normalizer(input_case="cased", cache_dir=os.path.join(os.getcwd(), "en_grammars"))
            sentences_norm = normalizer.normalize_list(
                sentences, verbose=False, punct_post_process=True, n_jobs=n_jobs, batch_size=batch_size
            )
            if len(sentences_norm) != len(sentences):
                raise ValueError("Normalization failed, number of sentences does not match.")
            else:
                sentences = sentences_norm
        return sentences

    # replace numbers with num2words
    def replace_num(sentences):
        sentences = '\n'.join(sentences)
        try:
            p = re.compile("\d+")
            new_text = ""
            match_end = 0
            for i, m in enumerate(p.finditer(sentences)):
                match = m.group()
                match_start = m.start()
                if i == 0:
                    new_text = sentences[:match_start]
                else:
                    new_text += sentences[match_end:match_start]
                match_end = m.end()
                new_text += sentences[match_start:match_end].replace(match, num2words(match, lang=language))
            new_text += sentences[match_end:]
            sentences = new_text
            sentences = re.sub(r' +', ' ', sentences)
            return sentences
        except NotImplementedError:
            print(
                f"{language} might be missing in 'num2words' package. Add required language to the choices for the"
                f"--language argument."
            )
            raise
    
    def final(sentences):
        sentences = sentences.lower()
        vocabulary_symbols = get_vocabulary_symbols(vocabulary)
        symbols_to_remove = ''.join(set(sentences).difference(set(vocabulary_symbols + ["\n", ""])))
        sentences = sentences.translate(''.maketrans(symbols_to_remove, len(symbols_to_remove) * " "))

        # remove extra space
        sentences_norm = re.sub(r' +', ' ', sentences)
        return sentences_norm

    with_punct = re.sub(r' +', ' ', "\n".join(sentences))
    sentences = substitute(sentences)
    punct_norm = replace_num(sentences)
    sentences_norm = final(with_punct)
    return {'': sentences_norm, '_with_punct_normalized': punct_norm, '_with_punct': with_punct}
