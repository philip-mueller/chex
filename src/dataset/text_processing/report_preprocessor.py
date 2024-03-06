import logging
import re
from typing import Dict, Optional, List, Sequence, Union, Tuple

import stanza
from stanza import Document

"""
Some pre-processing is taken from https://github.com/ttanida/rgrg
"""

from dataset.text_processing.section_parser import custom_mimic_cxr_rules, section_text

log = logging.getLogger(__name__)


def split_into_sections(full_text_report: str, study: str) -> Optional[Dict[str, str]]:
    # exclude these special cases
    custom_section_names, custom_indices = custom_mimic_cxr_rules()
    if study in custom_indices or study in custom_section_names:
        return None

    sections, section_names, section_idx = section_text(full_text_report)
    sections_by_name = {}
    for section, name in zip(sections, section_names):
        sections_by_name[name] = section

    return sections_by_name



PATTERN_REPLACE_MULTILINES = re.compile(r'(?:[\t ]*(?:\r\n|\n)+)+', flags=re.MULTILINE)
PATTERN_REPLACE_MULTISPACES = re.compile(r'[\t ]+')
PATTERN_REPLACE_MULTIDOT = re.compile(r'(\. ?)+')
SUBSTRINGS_TO_REMOVE = "WET READ VERSION|WET READ|UPRIGHT PORTABLE AP CHEST RADIOGRAPH:|UPRIGHT AP VIEW OF THE CHEST:|UPRIGHT AP AND LATERAL VIEWS OF THE CHEST:|TECHNOLOGIST'S NOTE:|TECHNIQUE:|SUPINE PORTABLE RADIOGRAPH:|SUPINE PORTABLE CHEST RADIOGRAPHS:|SUPINE PORTABLE CHEST RADIOGRAPH:|SUPINE PORTABLE AP CHEST RADIOGRAPH:|SUPINE FRONTAL CHEST RADIOGRAPH:|SUPINE CHEST RADIOGRAPH:|SUPINE AP VIEW OF THE CHEST:|SINGLE SUPINE PORTABLE VIEW OF THE CHEST:|SINGLE SEMI-ERECT AP PORTABLE VIEW OF THE CHEST:|SINGLE PORTABLE UPRIGHT CHEST RADIOGRAPH:|SINGLE PORTABLE CHEST RADIOGRAPH:|SINGLE PORTABLE AP CHEST RADIOGRAPH:|SINGLE FRONTAL VIEW OF THE CHEST:|SINGLE FRONTAL PORTABLE VIEW OF THE CHEST:|SINGLE AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|SINGLE AP UPRIGHT CHEST RADIOGRAPH:|SINGLE AP PORTABLE CHEST RADIOGRAPH:|SEMIERECT PORTABLE RADIOGRAPH OF THE CHEST:|SEMIERECT AP VIEW OF THE CHEST:|SEMI-UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT PORTABLE CHEST RADIOGRAPH:|SEMI-UPRIGHT PORTABLE AP RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT AP VIEW OF THE CHEST:|SEMI-ERECT PORTABLE FRONTAL CHEST RADIOGRAPH:|SEMI-ERECT PORTABLE CHEST:|SEMI-ERECT PORTABLE CHEST RADIOGRAPH:|REPORT:|PORTABLES SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE UPRIGHT FRONTAL VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE ABDOMEN:|PORTABLE SUPINE FRONTAL VIEW OF THE CHEST:|PORTABLE SUPINE FRONTAL CHEST RADIOGRAPH:|PORTABLE SUPINE CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT RADIOGRAPH:|PORTABLE SEMI-UPRIGHT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT AP CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPHS:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT AP AND PA CHEST RADIOGRAPH:|PORTABLE FRONTAL VIEW OF THE CHEST:|PORTABLE FRONTAL CHEST RADIOGRAPH:|PORTABLE ERECT RADIOGRAPH:|PORTABLE CHEST RADIOGRAPH:|PORTABLE AP VIEW OF THE CHEST:|PORTABLE AP UPRIGHT CHEST RADIOGRAPH:|PORTABLE AP CHEST RADIOGRAPH:|PA AND LATERAL VIEWS OF THE CHEST:|PA AND LATERAL CHEST RADIOGRAPHS:|PA AND LATERAL CHEST RADIOGRAPH:|PA AND LAT CHEST RADIOGRAPH:|PA AND AP CHEST RADIOGRAPH:|NOTIFICATION:|IMPRESSON:|IMPRESSION: AP CHEST:|IMPRESSION: AP|IMPRESSION:|IMPRESSION AP|IMPRESSION|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPPER ABDOMINAL RADIOGRAPH, TWO IMAGES:|FRONTAL SUPINE PORTABLE CHEST:|FRONTAL SEMI-UPRIGHT PORTABLE CHEST:|FRONTAL RADIOGRAPH OF THE CHEST:|FRONTAL PORTABLE SUPINE CHEST:|FRONTAL PORTABLE CHEST:|FRONTAL PORTABLE CHEST RADIOGRAPH:|FRONTAL LATERAL VIEWS CHEST:|FRONTAL LATERAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPHS:|FRONTAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPH WITH THE PATIENT IN SUPINE AND UPRIGHT POSITIONS:|FRONTAL AND LATERAL VIEWS OF THE CHEST:|FRONTAL AND LATERAL FRONTAL CHEST RADIOGRAPH:|FRONTAL AND LATERAL CHEST RADIOGRAPHS:|FRONTAL AND LATERAL CHEST RADIOGRAPH:|FRONTAL|FINIDNGS:|FINDNGS:|FINDINGS:|FINDINGS/IMPRESSION:|FINDINGS AND IMPRESSION:|FINDINGS|FINDING:|FINAL REPORT FINDINGS:|FINAL REPORT EXAMINATION:|FINAL REPORT|FINAL ADDENDUM ADDENDUM:|FINAL ADDENDUM ADDENDUM|FINAL ADDENDUM \*\*\*\*\*\*\*\*\*\*ADDENDUM\*\*\*\*\*\*\*\*\*\*\*|FINAL ADDENDUM|EXAMINATION: DX CHEST PORT LINE/TUBE PLCMT 1 EXAM|CONCLUSION:|COMPARISONS:|COMPARISON:|COMPARISON.|CHEST:|CHEST/ABDOMEN RADIOGRAPHS:|CHEST, TWO VIEWS:|CHEST, SINGLE AP PORTABLE VIEW:|CHEST, PA AND LATERAL:|CHEST, AP:|CHEST, AP UPRIGHT:|CHEST, AP UPRIGHT AND LATERAL:|CHEST, AP SUPINE:|CHEST, AP SEMI-UPRIGHT:|CHEST, AP PORTABLE, UPRIGHT:|CHEST, AP AND LATERAL:|CHEST SUPINE:|CHEST RADIOGRAPH:|CHEST PA AND LATERAL RADIOGRAPH:|CHEST AP:|BEDSIDE UPRIGHT FRONTAL CHEST RADIOGRAPH:|AP:|AP,|AP VIEW OF THE CHEST:|AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|AP UPRIGHT CHEST RADIOGRAPH:|AP UPRIGHT AND LATERAL CHEST RADIOGRAPHS:|AP PORTABLE SUPINE CHEST RADIOGRAPH:|AP PORTABLE CHEST RADIOGRAPH:|AP FRONTAL CHEST RADIOGRAPH:|AP CHEST:|AP CHEST RADIOGRAPH:|AP AND LATERAL VIEWS OF THE CHEST:|AP AND LATERAL CHEST RADIOGRAPHS:|AP AND LATERAL CHEST RADIOGRAPH:|5. |4. |3. |2. |1. |#1 |#2 |#3 |#4 |#5 "
def clean_section_text(text: str) -> str:
    text = remove_wet_read(text)
    text = re.sub(SUBSTRINGS_TO_REMOVE, "", text, flags=re.DOTALL)
    text = PATTERN_REPLACE_MULTISPACES.sub(' ', text).strip()
    return text


def remove_wet_read(text):
    """Removes substring like 'WET READ: ___ ___ 8:19 AM' that is irrelevant."""
    # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
    index_slices_to_remove = []
    for index in range(len(text)):
        if text[index:index + 8] == "WET READ":

            # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
            for curr_index in range(index + 8, len(text)):
                # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                # if the next WET READ substring is encountered
                if text[curr_index:curr_index + 2] in ["AM", "PM"] or text[curr_index:curr_index + 8] == "WET READ":
                    break

                # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                if text[curr_index:curr_index + 2] in ["AM", "PM"]:
                    index_slices_to_remove.append((index, curr_index + 2))

    # remove the slices in reversed order, such that the correct index order is preserved
    for indices_tuple in reversed(index_slices_to_remove):
        start_index, end_index = indices_tuple
        text = text[:start_index] + text[end_index:]

    return text


def clean_sentence(sentence: str) -> str:
    # remove newlines or multiple newlines (replace by space)
    sentence = PATTERN_REPLACE_MULTILINES.sub('\n', sentence)
    sentence = sentence.replace('\n', ' ')
    # merge multiple spaces into one
    sentence = PATTERN_REPLACE_MULTISPACES.sub(' ', sentence)
    # remove multiple dots (replace by one dot)
    sentence = PATTERN_REPLACE_MULTIDOT.sub('.', sentence)
    sentence = PATTERN_REPLACE_MULTISPACES.sub(' ', sentence).strip()
    # capitalize first letter
    if len(sentence) > 0:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence

def remove_duplicate_sentences(sentences: List[str]) -> List[str]:
    return list(dict.fromkeys(sentences))


class ReportProcessor:
    def __init__(self, lang: str = 'en', min_words: int = 2, section_names: Sequence[str] = ('findings', 'impression')):
        stanza.download(lang)
        self.stanza_tokenizer = stanza.Pipeline(lang, processors='tokenize', use_gpu=False)
        self.min_words = min_words
        self.section_names = section_names

    def __call__(self, report_full_text: str, study: str) -> Optional[List[str]]:
        sections = split_into_sections(report_full_text, study)
        if sections is None:
            log.warning(f"Ignoring study {study}.")
            return None
        if not any(key in sections for key in self.section_names):
            log.warning(f"Ignoring study {study} because it does not contain any of the following sections: {self.section_names}. Available sections: {sections.keys()}.")
            return None
        relevant_sections: List[str] = [sections[key] for key in self.section_names if key in sections]
        relevant_sentences: List[str] = [sentence for section in relevant_sections for sentence in self._process_section(section)]
        if len(relevant_sentences) == 0:
            log.warning(f"Ignoring study {study} because it does not contain any sentences in the relevant sections (or only too short or removed sentences).")
            return None
        return relevant_sentences

    def _process_section(self, section_txt: str) -> List[str]:
        section_txt = clean_section_text(section_txt)
        doc: Document = self.stanza_tokenizer(section_txt)

        sentences = [sent.text for sent in doc.sentences]
        sentences = [clean_sentence(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence.split()) >= self.min_words]
        sentences = remove_duplicate_sentences(sentences)

        return sentences
    

class SentenceProcessor:
    def __init__(self, min_words: int = 2):
        self.min_words = min_words

    def __call__(self, sentences: List[str]) -> List[str]:
        sentences = [self.process_sentence(sentence) for sentence in sentences]
        sentences = [sentence for sentence in sentences if len(sentence.split()) >= self.min_words]
        sentences = remove_duplicate_sentences(sentences)
        return sentences
    
    def process_sentence(self, text: str) -> str:
        text = remove_wet_read(text)
        text = re.sub(SUBSTRINGS_TO_REMOVE, "", text, flags=re.DOTALL)
        text = PATTERN_REPLACE_MULTISPACES.sub(' ', text).strip()
        text = clean_sentence(text)
        return text
