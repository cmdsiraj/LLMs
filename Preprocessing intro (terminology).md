# Tokenization
- It is a process of converting either Corpus(Paragraph) or Document(Sentences) in to ***"Tokens"***.
- When we apply **Tokenization** on **Corpus(Paragraph)**, then the **Corpus(Paragraph)** is converted into **Documents(Sentences)**
- When we apply **Tokenization** on **Document(Sentences)**, then the **Document(Sentences)** is converted into **Words**.
- Example:
  ```
  Corpus: "The belief of your mind is the thought of your mind. It is the seed from which your actions grow, influencing how you see yourself and the world around you."
  After applying tokenization, This corpus is converted to sentences. to convert to sentences, the corpus is breaked at special charaters like "." and "!".
  Sentences: 1. "The belief of your mind is the thought of your mind"
             2. "It is the seed from which your actions grow, influencing how you see yourself and the world around you"

  We can also apply tokenization to above sentences. when we apply tokenization to sentences, they gets converted to words. 1. "The", "belief", .....
                                                                                                                            2. "It", "is", "the", "seed",.....
  ```

- ### Vocabulary: Collection of all unique words in a Corpus is called vocabulary.

______________

# Stemming
- It is a process of reducing words to its base or root form, often by removing suffixes or preffixes.
- The goal is to treat different forms of word as a same item.
- Stemming is useful especially in tasks like **Search**, **Text Classification**, **Information Retrieval**.
- Example: [running, runs, ran, runner] ---> ran
- **Common Stemming Algorithms**
  1. **Porter Stemmer** -  Most widely used, especially in English
  2. **Snowball Stemmer** - An improvement over porter.
  3. **Lancaster Stemmer** - More aggressive, may remove too much.
- **Disadvantages**: Sometimes stemming produces words that have no meaning (non-dictionary words).
- To overcome this disadvantage, we can use **Lemmetaization**.

# Lemmatization
- It is a technique like stemming.
- The output we will get after lemmatization is called ***'lemma'***, which is a ***root word*** rather than ***root stem***, the output of stemming.
- After lemmatization, we will be getting ***a valid word that means the same thing***.
> NLTK provides `WordNetLemmatizer` class which is a thin wrapper around the ***wordnet corpus***. This class uses `morphy()` function to the `WordNet CorpusReader` class to find lemma.
- Since, we are looking for a root word from Wordnet Corpus, `WordNetLemmatizer` takes some time.
- Examples where Lemmatization is useful: Q&A, chatbots, text summerization etc.,

# StopWords
- ***stopwords*** are common words that are usually ***filtered out*** before processing text because they carry ***little meaningful information*** for tasks like search, classification, or clustering.
- **Removing stopwords:**
  - Reduces noise in text data
  - Speeds up processing
  - Improves focus on informative words (e.g., nouns, verbs)
