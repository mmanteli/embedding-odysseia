# Text embedding model selection reasoning 


We want to maximize *closeness in space*. MTEB measures this with these benchmarks (checked):
- [ ] Classification (12 tasks, contains multilingual tasks) <
- [x] Clustering (11 tasks)
- [x] Sentence Pair classification (yes-no same meaning, 3 tasks)
- [ ] Reranking (rank a list of documents based on relevance, 4 tasks)
- [ ] Retrieval (find relevant docs in a corpus, 15 tasks)
- [x] Semantic Text Similarity (STS) (10 tasks, contains multilinguality)
- [ ] Summarisation (1 task)
- [ ] Bitext mining (2(?) tasks, built-in multilinguality)


 
# From [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)


> This is infact MMTEB ranking. On the leaderboard, you can choose which bencmark to use. These results are from MTEB (multilingual, v1) which should equal MMTEB.
> Also they literally updated the layout of this page as I was using it 🙄

## 2nd Overall, best in focus tasks and Mean (TaskType) [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)

Embedding size is **3584**, model size 7B
### PROS
- large sequence length of 32k (conflicting numbers reported)
- best in **multilabel classification and pair classification**
- best in reranking
- best wrt. Mean (TaskType) (I gues each tasktype weighted the same, since there are different number of tasks per category)
### CONS
- Ranks best specifically on English and Chinese
- Other languages included though, e.g. TurkuNLP Finnish Paraphrase Corpus is in MMTEB, on which this model is 9th best
	- bge-m3 is the best on Finnish, 30.34%, this model gets 25.72%

### OTHER
I looked at the top20 models (excluding this) and this performs much better than the average 20 models except has WAY worse performance on these tasks:
- Indic Language identification
- Cyrillic Turkic language Classification
These are like, 18-34%-points lower. Otherwise, either better, or 1-2%-points lower. Financial and patent stuff is also 5%-points lower, but we don't care, huh? (/j)

 

## 1st Overall [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)

Built on the foundations of [E5-mistral-7b-instruct ](https://huggingface.co/intfloat/e5-mistral-7b-instruct) and [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1).
Embedding size is **4096**, 7B parameters, sequence length size is 32k
### PROS
- retrieval is good according to authors, not the best in retreival on leaderboard but good values nonetheless
- if on top of E5, I would assume it has multilinguality (at least Korean in the examples)
### CONS
- Retrieval is not the thing we want

 
## 3rd Overall, best Mean(Task) [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) 
#recommended
 
Embedding size is **1024**. Best on classification and Bitext mining.
### PROS
- others use it $\rightarrow$ help is available
- from XLM-RoBERTa-base $\rightarrow$ small size
- multilinguality
- **Better in STS than the two above** 
### CONS
- small sequence length of 512+2, which might actually be a large drawback
- Not best in the interesting tasks

### OTHER
I also looked at this vs. average of 20 "best" models. Similarly to gte-Qwen2, mostly better results, maybe this had a bit more "red" (less than average) but the values were small. This also had some large drops in some categories, like Faroese for example, but those were few.

 
## 7th overall [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 
#recommended 

Embedding size is **4096**. Sequence length is 4096 as well.
### PROS
- Like above, but with larger sequence length
### CONS
- Has some multilingual capability but is designed for English

 
## 19th Overall, second best in STS  [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) 
#recommended 

Sequence length up to 8192. Based on XLM-R (with other finetuning steps in between)
### PROS
- Multiple possible embedding sizes (Matryoshka embedding!)
- Multiple possible embedding types, e.g. separate for clustering and STS and querying
- Support 30 languages explicitly (**Arabic, Bengali, Chinese, Danish, Dutch, English, Finnish, French, Georgian, German, Greek, Hindi, Indonesian, Italian, Japanese, Korean, Latvian, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Thai, Turkish, Ukrainian, Urdu,** and **Vietnamese.**) and may inherit other languages from XLM-R
### CONS
- overall performance is not that good, 5-6 %-points below the lead in Mean (Task) and Mean (TaskType)

# Conclusion

Starting with multilingual-e5-instruct just to get it working. Then moving to gte-Qwen or JinaAi