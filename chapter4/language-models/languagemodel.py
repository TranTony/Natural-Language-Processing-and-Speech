#Read file 
import re
import math
from itertools import product

start_word = "<s>"
end_word = "</s>"

def readData(file_path):
	with open(file_path) as file: 
		 	return [re.split("\s+", line.rstrip('\n')) for line in file]

def calculate_Num_UniGram(sentences):
	unigram_count = 0
	for s in sentences:
		unigram_count += len(s) - 2
	return unigram_count



def calculate_NUM_BiGrame(sentences):
	bigram_count = 0;
	for s in sentences:
		bigram_count += len(s) - 1
	return bigram_count

class UniGramLM:
	def __init__(self, sentences, smoothing=False):
		self.uni_freq = dict()
		self.corpus_len = 0
		for s in sentences:
		 	for w in s:
		 		self.uni_freq[w] = self.uni_freq.get(w, 0) + 1
		 		#print(self.uni_freq[w])
		 		if w != start_word and w != end_word:
		 			self.corpus_len += 1
		self.num_unique_words = len(self.uni_freq) - 2

	# unigram = count(word)/count(words)
	def prob_unigram(self, word):
		num_word = self.uni_freq.get(word, 0)
		num_corpus_word = self.corpus_len
		return float(num_word) / float(num_corpus_word)


	# sentence prob = multiple of prob word unigram P(a b c) = P(a) * P(b) * P(c)
	def sentence_prob(self, sentence, normalize_prob=True):
		log_sum_prob_sen = 0
		prob_sen = 1
		for w in sentence:
			word_prob = self.prob_unigram(w)
			prob_sen *= word_prob
		log_sum_prob_sen = math.log(prob_sen, 2)
		return math.pow(2, log_sum_prob_sen) if normalize_prob else log_sum_prob_sen

class NGram:
	def __init__(self, sentences, nGram, smoothing=False):
		self.N_freq = dict()
		self.N_1_freq = dict()
		self.N_Prob = dict()
		self.N_freq = self.NGram_count(sentences, nGram)
		self.N_1_freq = self.NGram_count(sentences, nGram - 1)
		self.N_Prob = self.NGram_prob(nGram);

	# Counting the number of n words
	def NGram_count(self, sentences, nGram):
		N_freq = dict()
		for s in sentences:
			previous_words = ()
			pr = tuple(s)
			n_count = 0
			for index in range(len(pr) + 1 - nGram):
				n_count = index
				for ind in range(nGram):
					previous_words += (pr[n_count + ind],)

				if len(previous_words) == nGram:
					N_freq[(previous_words)] = N_freq.get((previous_words),0) + 1
					previous_words = ()
		return N_freq;

	# Estimate probability of NGram
	def NGram_prob(self, nGram):
		for  k, v in self.N_freq.items():
			for k_1, v_1 in self.N_1_freq.items():
				for index in range(len(k) + 1  - nGram):
					if k_1 == k[index:index + nGram - 1]:
						#print("P" + str(k) + "= " + str(float(v/v_1)))
						self.N_Prob[k] = float(v/v_1)
						break
		return self.N_Prob

	# Estimate log probability of a sentence
	def NGram_sentence_prob_log(self, nGram, sentence):
		previous_words = ()
		pr = sentence.split(" ")
		pr = tuple(pr)
		n_count = 0
		s_freq = []
		prob_sum_log = 0.0

		for index in range(len(pr) + 1 - nGram):
			n_count = index
			for ind in range(nGram):
				previous_words += (pr[n_count + ind],)

			if len(previous_words) == nGram:
				s_freq.append(previous_words)
				previous_words = () 

		for n_key in s_freq:
			try:
				prob_sum_log += math.log(self.N_Prob.get(n_key, 2))		
			except:
				prob_sum_log += float(0)
		return prob_sum_log
	
	# Estimate prob of a sentence
	def NGram_sentence_prob(self, nGram, sentence):
		previous_words = ()
		pr = sentence.split(" ")
		pr = tuple(pr)
		n_count = 0
		s_freq = []

		for index in range(len(pr) + 1 - nGram):
			n_count = index
			for ind in range(nGram):
				previous_words += (pr[n_count + ind],)

			if len(previous_words) == nGram:
				s_freq.append(previous_words)
				previous_words = () 

		prob_mul = 1.0
		for n_key in s_freq:
			prob_mul *= self.N_Prob.get(n_key, 1)	

		return prob_mul

	# perplexity references to chapter 4 Language Modeling with N-grams.
	def NGram_perplexity(self, biGramML, sentence, nGram=2):
		prob_w = biGramML.NGram_sentence_prob(nGram, sentence)
		n_gram_count = len(sentence.split(" ")) - 2
		
		try:
			sentence_probability_log_sum = math.log(prob_w)
		except:
			sentence_probability_log_sum = float('-inf')
		
		exp = -1 / n_gram_count
		return math.pow(prob_w, exp)


if __name__ == '__main__':

	# corpus for Language Model
	data = readData("./sampledata.txt")

	# Test sentence
	sentence = "<s> I am a man I do not like </s>"

	# Number of Ngram
	N_Gram = 2

	# Language Models
	nGramML = NGram(data, N_Gram, smoothing=False)
	biGramML = NGram(data, 2, smoothing=False)

	# Estimate probability for language model
	nGramML.NGram_prob(N_Gram)

	# Calculate perplexity of the test sentence
	print(nGramML.NGram_perplexity(biGramML, sentence))

	# Calculate the probility of a sentence
	print("probability of sentence = " + str(nGramML.NGram_sentence_prob_log(N_Gram, sentence)))

	
				






