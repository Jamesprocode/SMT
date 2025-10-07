from typing import Literal
import os
import re

import numpy as np

from os import path
from loguru import logger

def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

@logger.catch
def check_and_retrieveVocabulary(YSequences, pathOfSequences, nameOfVoc):
    w2ipath = pathOfSequences + "/" + nameOfVoc + "w2i.npy"
    i2wpath = pathOfSequences + "/" + nameOfVoc + "i2w.npy"

    w2i = []
    i2w = []

    if not path.isdir(pathOfSequences):
        os.mkdir(pathOfSequences)

    if path.isfile(w2ipath):
        w2i = np.load(w2ipath, allow_pickle=True).item()
        i2w = np.load(i2wpath, allow_pickle=True).item()
    else:
        w2i, i2w = make_vocabulary(YSequences, pathOfSequences, nameOfVoc)

    return w2i, i2w

def make_vocabulary(YSequences, pathToSave, nameOfVoc):
    vocabulary = set()
    for samples in YSequences:
        for element in samples:
                vocabulary.update(element)

    #Vocabulary created
    w2i = {symbol:idx+1 for idx,symbol in enumerate(vocabulary)}
    i2w = {idx+1:symbol for idx,symbol in enumerate(vocabulary)}
    
    w2i['<pad>'] = 0
    i2w[0] = '<pad>'

    #Save the vocabulary
    np.save(pathToSave + "/" + nameOfVoc + "w2i.npy", w2i)
    np.save(pathToSave + "/" + nameOfVoc + "i2w.npy", i2w)

    return w2i, i2w

@logger.catch
def save_kern_output(output_path, array):
    for idx, content in enumerate(array):
        transcription = "".join(content)
        transcription = transcription.replace("<t>", "\t")
        transcription = transcription.replace("<b>", "\n")
    
        with open(f"{output_path}/{idx}.krn", "w") as bfilewrite:
            bfilewrite.write(transcription)

def fix_kern_terminators(krn: str) -> str:
    """Ensure all spines are properly terminated with *-"""
    lines = krn.strip().split('\n')
    if not lines:
        return krn

    # Find number of spines from first data line
    num_spines = None
    for line in lines:
        if line.strip() and not line.startswith('!!!'):
            num_spines = len(line.split('\t'))
            break

    if num_spines is None:
        return krn

    # Check last line
    last_line = lines[-1].strip()
    expected_terminator = '\t'.join(['*-'] * num_spines)

    if last_line == expected_terminator:
        return krn
    elif '*-' in last_line:
        # Fix incomplete terminators
        terminators = last_line.split('\t')
        while len(terminators) < num_spines:
            terminators.append('*-')
        lines[-1] = '\t'.join(terminators[:num_spines])
    else:
        # Add missing terminator line
        lines.append(expected_terminator)

    return '\n'.join(lines)

def clean_kern(
        krn: str,
        forbidden_tokens: list[str] = ["*staff2", "*staff1", "*Xped", "*tremolo", "*ped", "*Xtuplet", "*tuplet", "*Xtremolo", "*cue", "*Xcue", "*rscale:1/2", "*rscale:1", "*kcancel", "*below"]
        ) -> str:
    forbidden_pattern = "(" + "|".join([t.replace("*", "\\*") for t in forbidden_tokens]) + ")"
    krn = re.sub(f".*{forbidden_pattern}.*\n", "", krn) # Remove lines containing any of the forbidden tokens
    krn = re.sub("(^|(?<=\n))\*(\s\*)*(\n|$)", "", krn) # Remove lines that only contain "*" tokens
    # Fix terminators before returning
    return fix_kern_terminators(krn.strip())

def parse_kern(
        krn: str,
        krn_format: Literal["standard"] | Literal["kern"] | Literal["ekern"] | Literal["bekern"] = "bekern" 
        ) -> list[str]:
    krn = clean_kern(krn)
    krn = re.sub("(?<=\=)\d+", "", krn)

    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")

    if krn_format == "kern":
        krn = krn.replace("·", "").replace('@', '')
    elif krn_format == "ekern":
        krn = krn.replace("·", " ").replace('@', '')
    elif krn_format == "bekern":
        krn = krn.replace("·", " ").replace("@", " ")

    return krn.strip().split(" ")
