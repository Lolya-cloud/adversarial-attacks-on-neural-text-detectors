import nltk
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_synonyms(word, pos):
    synonyms = []
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def rephrase_words(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    rephrased_text = []

    for word, pos in tagged:
        # tags for adjectives, nouns, and verbs
        wordnet_pos = {'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
                       'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
                       'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB,
                       'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB}
        if pos in wordnet_pos:
            synonyms = get_synonyms(word, wordnet_pos[pos])
            if synonyms:
                rephrased_text.append(synonyms[0])  # replace with the first synonym
            else:
                rephrased_text.append(word)  # if no synonym found, keep the original word
        else:
            rephrased_text.append(word)

    return ' '.join(rephrased_text)

text = """In today's fast-paced world, technology has become an integral part of our lives. From smartphones to artificial intelligence, we rely on various technological advancements to make our lives easier and more convenient. Communication has also been greatly transformed by these innovations. Social media platforms have connected people from all corners of the globe, enabling instant communication and fostering virtual communities.
However, with the increasing reliance on technology, concerns about privacy and security have also emerged. Data breaches and cyber attacks have become common occurrences, raising questions about the protection of personal information. As a result, individuals and organizations are becoming more vigilant about safeguarding their digital identities and taking steps to enhance cybersecurity.
Despite the challenges, technology continues to evolve and shape our future. It has revolutionized industries such as healthcare, transportation, and entertainment, bringing about significant improvements in efficiency, accessibility, and overall quality of life. Artificial intelligence and machine learning are being applied in various fields, including medical diagnosis, autonomous vehicles, and personalized recommendations.
As we move forward, it is crucial to strike a balance between embracing technological advancements and addressing the ethical and social implications they bring. With proper regulation and responsible usage, technology has the potential to empower individuals and societies, driving progress and innovation for generations to come."""
tags_adj = ['JJ', 'JJR', 'JJS']
tags_noun = ['NN', 'NNS', 'NNP', 'NNPS']
tags_verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

print(rephrase_words(text))