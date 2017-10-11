NEWLINE='\n'

class ProbabilisticTagger():

    def __init__(self):
        self.tagger_dict = dict()
        self.unigram_tag_dict = dict()
        self.bigram_tag_dict = dict()
        self.unigram_word_dict = dict()
        self.bigram_word_dict = dict()
        self.transition_tag_dict = dict()
        self.emission_prob_dict = dict()
        self.list_of_invalid_lines = [NEWLINE]
        self.viterbi = dict()
        self.backtrack = dict()
        self.unk_list = list()

    def mapWordToTags(self, training_file):
        fh = open(training_file, 'r')
        content = fh.readlines()
        
        for line in content:
            if line in self.list_of_invalid_lines:
                continue
            
            line = line.strip()
            list_line = line.split('\t')
            
            word = list_line[1]
            if word in self.unk_list:
                word = "UNK"
            tag  = list_line[2]
            
            if word not in self.tagger_dict:
                self.tagger_dict[word] = dict()
                self.tagger_dict[word][tag] = 1
            else:
                if tag not in self.tagger_dict[word]:
                    self.tagger_dict[word][tag] = 1
                else:
                    self.tagger_dict[word][tag] += 1

        fh.close()

    def __getTagForWord(self, word):
        if word not in self.tagger_dict:
            return "PRP"
        else:
            return max(self.tagger_dict[word], key=self.tagger_dict[word].get)

    def __populate_unigram_tag_dict (self, unigram_tag_file):
        fh = open(unigram_tag_file, "r")
        content = fh.readlines()

        for line in content:
            count = line.split()[0]
            tag = line.split()[1]
            self.unigram_tag_dict[tag] = count

        fh.close()

    def createTransitionProbability(self, bigram_file, unigram_file):
        self.__populate_unigram_tag_dict(unigram_file)
        fh = open(bigram_file, "r")
        content = fh.readlines()
        
        
        for line in content:
            bigram_elem = line.strip().split()
            count = int(bigram_elem[0])
            bigram_string = bigram_elem[1] + "," + bigram_elem[2]
            self.bigram_tag_dict[bigram_string] = count

                
        for u1 in self.unigram_tag_dict:
            for u2 in self.unigram_tag_dict: 
                if u1 not in self.transition_tag_dict:
                    self.transition_tag_dict[u1] = dict()

                bigram_str = u1 + "," + u2
                if bigram_str not in self.bigram_tag_dict:
                    bigram_str_value = 0
                else:
                    bigram_str_value = self.bigram_tag_dict[bigram_str]

                self.transition_tag_dict[u1][u2] = bigram_str_value/float(self.unigram_tag_dict[u1])
        fh.close()

    def createEmissionProbability(self):
        for word in self.tagger_dict:
            pos_list = self.tagger_dict[word].keys()
            self.emission_prob_dict[word] = dict()
            for pos in pos_list:
                self.emission_prob_dict[word][pos] = self.tagger_dict[word][pos]/float(self.unigram_tag_dict[pos])

    def mapTestDataToTags(self, test_file_untagged, test_file_tagged):
        fh = open (test_file_tagged, 'w')
        fh1 = open(test_file_untagged, 'r')
        
        lines = fh1.readlines()
        for line in lines:
     
            if line in self.list_of_invalid_lines:
                continue
            
            line = line.strip()
            list_line = line.split('\t')
            index = list_line[0]
            word = list_line[1]
            probTag = self.__getTagForWord (word)
            end_delim = '\n'
            if word == '.':
                end_delim += '\n'
            
            lineWithTag = index + '\t' + word + '\t' + probTag + end_delim
            fh.write(lineWithTag)
           
        fh1.close()
        fh.close()

    def __createUnkTags(self):
        self.unk_list = [k for (k,v) in self.unigram_word_dict.iteritems() if v == 1]
        #print self.unk_list
        ## Deleted all the Unigram words with count 1 in the unigram word dict
        ## And then replaced all these words with a single UNK word with the 
        ## count equal to the length of the words which have occurred only once
        for u in self.unk_list:
            del self.unigram_word_dict[u]
        self.unigram_word_dict["UNK"] = len(self.unk_list)

         
    def __populate_bigram_word_dict (self, word_bigram_file):
        fh = open(word_bigram_file, 'r')
        content = fh.readlines()
        for line in content:
            bigram_list = line.strip().split()
            bigram1 = bigram_list[1]
            if bigram1 in self.unk_list:
                bigram1 = "UNK"
            bigram2 = bigram_list[2]
            if bigram2 in self.unk_list:
                bigram2 = "UNK"

            bigram_string = bigram1 + "," + bigram2 
            self.bigram_word_dict[bigram_string] = int(bigram_list[0])

        fh.close()

    def __get_tag_set (self, test_words):
        tag_list = list()
        tag_set = set()
        for word in test_words:
            tag_list_for_word = self.tagger_dict[word].keys()
            tag_list.extend(tag_list_for_word)

        tag_set = set(tag_list)
        tag_set.add(".")
        return tag_set

    def __get_updated_test_words(self, test_words):
        updated_test_words = list()
        for word in test_words:
            if word not in self.unigram_word_dict:
                updated_test_words.append("UNK")
            else:
                updated_test_words.append(word)
        
        return updated_test_words

    def run_viterbi(self, test_sent):
        test_words = self.__get_updated_test_words(test_sent)
        tag_set = self.__get_tag_set (test_words)
        start_state = '.'
        end_state = '.'
        
        ## Initialisation of Viterbi done
        for tag in tag_set:
            self.viterbi[tag] = dict()
            self.backtrack[tag] = dict()
            if tag not in self.emission_prob_dict[test_words[0]]:
                emission_prob_value = 0
            else:
                emission_prob_value = self.emission_prob_dict[test_words[0]][tag]

            self.viterbi[tag][0] = self.transition_tag_dict[start_state][tag] * emission_prob_value
            self.backtrack[tag][0] = [-1, -1] 
        
        for i in range(1, len(test_words)):
            for tag in tag_set:
                maxm_vit = 0
                maxm_vit_tag = 0
                maxm_back = 0
                maxm_back_tag = 0
                if tag not in self.emission_prob_dict[test_words[i]]:
                    emission_prob_val = 0
                else:
                    emission_prob_val = self.emission_prob_dict[test_words[i]][tag]

                for tag1 in tag_set:
                    vit_val = self.viterbi[tag1][i-1] * self.transition_tag_dict[tag1][tag] * emission_prob_val 
                    if vit_val > maxm_vit:
                        maxm_vit = vit_val
                        maxm_vit_tag = tag1

                    back_val = self.viterbi[tag1][i-1] * self.transition_tag_dict[tag1][tag]
                    if back_val > maxm_back:
                        maxm_back = back_val
                        maxm_back_tag = tag1

                self.viterbi[tag][i] = maxm_vit
                self.backtrack[tag][i] = [maxm_back_tag, i-1]

        maxm_vit = 0
        maxm_back_tag = 0
        for tag in tag_set:
            vit_final_val = self.viterbi[tag][len(test_words)-1] * self.transition_tag_dict[tag][end_state]
            if vit_final_val > maxm_vit:
                maxm_vit = vit_final_val
                maxm_back_tag = tag

        self.viterbi[end_state][len(test_words)-1] = maxm_vit
        self.backtrack[end_state][len(test_words)-1] = [maxm_back_tag, len(test_words)-1]

        final_tag_list = list()
        final_tag_list.append(maxm_back_tag)
        backtrack_list = self.backtrack[maxm_back_tag][len(test_words)-1]
        
        iter_length = len(test_words) - 1
        while iter_length > 0:
            final_tag_list.append(backtrack_list[0])
            backtrack_list = self.backtrack[backtrack_list[0]][backtrack_list[1]]
            iter_length -= 1

        #print final_tag_list[::-1]
        return final_tag_list[::-1]
        
    def __populate_unigram_word_dict (self, word_unigram_file):
        fh = open(word_unigram_file, 'r')
        content = fh.readlines()
        for line in content:
            unigram_list = line.strip().split()
            self.unigram_word_dict[unigram_list[1]] = int(unigram_list[0])
        fh.close()

    def populateWordInfo (self, word_bigram_file, word_unigram_file):
        self.__populate_unigram_word_dict (word_unigram_file)
        self.__createUnkTags()
        self.__populate_bigram_word_dict (word_bigram_file)


    def run_testcase(self, test_untagged_file, viterbi_tagged_file):
        fh = open(test_untagged_file, "r")
        fh1 = open (viterbi_tagged_file, 'w')
        
        lines = fh.readlines()
        temp_list = list()
        sentence_list = list()
        for line in lines:
            if line in self.list_of_invalid_lines:
                sentence_list.append(temp_list)
                temp_list = []
            else:
                toks = line.split()
                temp_list.append(toks[1])

        for sent in sentence_list:
            output_tags = self.run_viterbi(sent[:len(sent)-1])
            file_entry = ''
            for i in range(0, len(sent)-1):
                file_entry += str(i+1) + "\t" + sent[i] + "\t" + output_tags[i] + "\n"
            file_entry += str(len(sent)) + "\t" + "." + "\t" + "." + "\n" + "\n"
            fh1.write (file_entry) 
        

    def addKSmoothing (self):
        for ftag in self.transition_tag_dict:
            stag_list = self.transition_tag_dict[ftag]
            for stag in stag_list:
                bigram_str = ftag+","+stag
                if bigram_str not in self.bigram_tag_dict:
                    bigram_str_val = 0
                else:
                    bigram_str_val = self.bigram_tag_dict[bigram_str]
                self.transition_tag_dict[ftag][stag] = (bigram_str_val + 0.75)/(int(self.unigram_tag_dict[ftag]) + 0.75 * len(self.bigram_tag_dict)) 

if __name__ == "__main__":
    pt = ProbabilisticTagger()
    pt.populateWordInfo("word_bigram.txt", "word_unigram.txt")
    pt.mapWordToTags("berp-POS-training.txt") #Unknown (UNK) words handled here
    pt.createTransitionProbability ("tag_bigram.txt", "tag_unigram.txt")
    pt.createEmissionProbability ()
    pt.addKSmoothing()
    pt.run_testcase("assgn2-test-set.txt", "assgn2-test-set-tagged.txt")
    #pt.mapTestDataToTags("berp-POS-test-untagged.txt", "berp-POS-test-tagged.txt") #Used for baseline model
    #test_sentence = "i 'd like to go a fancy restaurant" 
    #print pt.emission_prob_dict
    #print pt.transition_tag_dict
    #pt.run_viterbi(test_sentence)
    #pt.run_testcase("berp-POS-test-untagged.txt", "berp-POS-viterbi-tagged.txt")
    #print pt.tagger_dict
    #print pt.unigram_tag_dict
    #print pt.emission_prob_dict
    #print pt.unigram_word_dict
    #print len(pt.unk_list)
    #print pt.bigram_word_dict
    #print pt.bigram_tag_dict
    #print pt.unigram_tag_dict
