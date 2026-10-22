"""HW4 Problem 2: Compression (numerical).

Run one part at a time (or all of them) from the command line, e.g.:

    python compression.py --part b
    python compression.py --part all

Data files:
    data/letter_frequencies.csv  English letter frequencies, from Wikipedia
                                 (https://en.wikipedia.org/wiki/Letter_frequency).
    data/sample.txt              10^5 letters drawn independently from that table.
    data/relativity.txt          Albert Einstein, "Relativity: The Special and
                                 General Theory" (trans. R. W. Lawson), Project
                                 Gutenberg eBook #5001.

Fill in the TODOs below.
"""

import argparse
import lzma
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def load_frequencies():
    """Letter probabilities p(x) from the table, as a dict {letter: p}.

    The frequencies in the table are rounded and sum to 1.00154, so we
    normalize them.
    """
    p = {}
    for line in (DATA_DIR / "letter_frequencies.csv").read_text().splitlines()[1:]:
        letter, frequency = line.split(",")
        p[letter] = float(frequency)
    total = sum(p.values())
    return {letter: p[letter] / total for letter in p}


def load_text(name):
    """A text file, upper-cased, with everything except A-Z removed."""
    text = (DATA_DIR / name).read_text(encoding="utf-8").upper()
    return "".join(c for c in text if c in LETTERS)


def save_bits(bits, name):
    """Write a string of bits such as "0110..." to output/<name>.bin.

    The bits are packed 8 to a byte (the last byte is padded with zeros).
    Returns the size of the file in bytes.
    """
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / (name + ".bin")
    path.write_bytes(np.packbits(np.array(list(bits), dtype=np.uint8)).tobytes())
    return path.stat().st_size


def huffman_code(p):
    """Huffman code for the distribution p, as a dict {letter: codeword},
    e.g. {"E": "100", ...}."""
    # TODO


def part_a():
    p = load_frequencies()

    # TODO


def part_b():
    p = load_frequencies()

    # TODO


def part_c():
    p = load_frequencies()

    # TODO


def part_d():
    p = load_frequencies()
    text = load_text("sample.txt")
    N = len(text)
    print("sample.txt: N = %d letters" % N)

    # Example: 5-bit code.
    five_bit = {x: format(i, "05b") for i, x in enumerate(LETTERS)}
    bits = "".join(five_bit[x] for x in text)
    size = save_bits(bits, "sample_5bit")
    print("  5-bit       %7.1f kB  %.3f bits/letter" % (size / 1e3, 8 * size / N))

    # TODO: Other encodings. One general-purpose compressor is lzma.compress(...).


def part_e():
    p = load_frequencies()
    text = load_text("relativity.txt")
    N = len(text)

    # TODO


def main():
    """Command line entry: DO NOT MODIFY"""
    parser = argparse.ArgumentParser(
        description="HW4 Compression -- run one part, or all of them."
    )
    parser.add_argument(
        "--part", choices=["a", "b", "c", "d", "e", "all"], required=True,
        help="Which part of the problem to run.",
    )
    args = parser.parse_args()

    dispatch = {"a": part_a, "b": part_b, "c": part_c, "d": part_d,
                "e": part_e}

    if args.part == "all":
        for func in dispatch.values():
            func()
    else:
        dispatch[args.part]()


if __name__ == "__main__":
    main()
