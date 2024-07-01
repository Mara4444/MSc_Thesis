# Exploring multilingual instruction understanding in zero-shot chain-of-thought reasoning

## Abstract

LLMs have demonstrated the ability to perform a range of tasks by virtue of understanding the task instructions provided in a form of a prompt. However, little research focuses on their ability to understand such instructions in different languages. Closing this gap, we systematically study how the zero-shot instruction understanding abilities of LLMs differ across languages. Specifically, we conduct a series of experiments on the Bloomz and Llama-2 model to evaluate their zero-shot performance when presenting English instructions versus machine-translated native instructions to the models. We comprehensively study this effect across three tasks and 55 languages. For many languages, the models fail to understand native instructions, but consistently exhibits a higher performance on data points from that language when English instructions are provided instead. Additional experiments show that chain-of-thought reasoning effectively enhances the performance on arithmetic reasoning tasks with Llama-2, however no particular trend can be identified for which languages benefit from English instructions.

## Experiments

### Datasets
XCOPA, XStorycloze, BNLI, MGSM and MSVAMP

### Large language models

- Llama-2-7b-chat
- Bloomz-7b1-mt

### Target languages

A set of 55 languages other than English is selected for the experiments. The experimental language set is: Afrikaans, Arabic, Armenian, Balinese, Belarussian, Bengali, Bosnian, Bulgarian, Burmese, Cantonese, Catalan, Chinese, Croatian, Czech, Danish, Dutch, Estonian, Finnish, French, German, Greek, Haitian Creole, Hebrew, Hindi, Hungarian, Indonesian, Italian, Javanese, Japanese, Khmer, Korean, Lao, Maithili, Malay, Malayam, Marathi, Nepali, Norwegian, Polish, Portuguese, Quechuan, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Telugu, Thai, Tibetan, Turkish, Ukrainian, Urdu, Vietnamese and Zulu.


## File structure

- bin: old files
- EDA: all files and code for the EDA milestone
- code: all code files for snellius and response notebooks
    - src: code source files
- datasets: machine translated datasets
- figures: saved plots for thesis
- lang2vec: used for URIEL language similarity
- results: response csv files
  
