from collections import Counter
from re import split


def main():
    counter = Counter()
    with open("story.txt", "r") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            counter.update(x for x in split("[^a-z]+", line) if x)
    freqSorted = sorted(counter, key=counter.get, reverse=True)
    longestWord = max(counter, key=len)
    wordLength = len(longestWord)

    count = 0
    print("Word".rjust(16), "|", "Count".ljust(16))
    print("=" * 35)
    for word in freqSorted:
        print(word.title().rjust(16), "|", str(counter[word]).ljust(16))
        count += 1
        if count >= 100:
            break


if __name__ == '__main__':
    main()
