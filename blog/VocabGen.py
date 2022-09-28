from numpy.core.numeric import full


def vocabexecute(full_text,type_of_grammar):
    import re
    import torch
    from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM

    # Load pre-trained model tokenizer (vocabulary)
    import time
    start = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()
    end = time.time()
    print ("Time Elapsed to load BERT ",end-start)

    # Main function to predict the top 30 choices for the fill in the blank word using BERT. 
    # Eg: The Sun is more ____ 4 billion years old.

    
    def get_predicted_words(text):
        text = "[CLS] " + text.replace("____", "[MASK]") + " [SEP]"
        # text= '[CLS] Tom has fully [MASK] from his illness. [SEP]'
        tokenized_text = tokenizer.tokenize(text)
        #print("tokenized sentence: ",tokenized_text,"\n")
        masked_index = tokenized_text.index('[MASK]')
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        # Get 30 choices for the masked(blank) word 
        k = 30
        predicted_index, predicted_index_values = torch.topk(predictions[0, masked_index], k)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index_values.tolist())
        filtered_tokens_to_remove_punctuation = []
        # Remove any predictions that contain punctuation etc as they are not relevant to us.
        for token in predicted_tokens:
            if re.match("^[a-zA-Z0-9_]*$", token):
                filtered_tokens_to_remove_punctuation.append(token)
            
        return filtered_tokens_to_remove_punctuation

    sentence = "They all look tiny ____ they are so far away from the Earth."
    print ("original sentence: ",sentence,"\n")
    predicted_words = get_predicted_words(sentence)
    print ("predicted choices: ", predicted_words)

    '''file_path = "egypt.txt" #other texts in same directory: "PSLE.txt", "hellenkeller.txt", "Grade7_electricity.txt" , "material.txt", "paperboat.txt"

    def read_file(file_path):
        with open(file_path, 'r') as content_file:
            content = content_file.read()
            return content
        
    text = read_file(file_path)'''
    text = full_text
    print(text)

    #  We will extract some adpositions. An adposition is a cover term for prepositions and postpositions.
    import pke
    import string

    def get_adpositions_multipartite(text):
        out=[]

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        #    not contain punctuation marks or stopwords as candidates.
        grammar_tags = {'Preposition':'ADP','Verb':'VERB','Noun':'NOUN','Conjunction':'CCONJ','Adjective':'ADJ','Determiner':'DET'}
        pos = set()
        if len(type_of_grammar)>0:
            for e in type_of_grammar:
                pos.add(grammar_tags[e])
        else:
            pos = {'ADP','NOUN','VERB','CCONJ','ADJ','DET'} #Adpositions
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=0.5,
                                    threshold=0.00,
                                    method='average')
        keyphrases = extractor.get_n_best(n=30)

        for key in keyphrases:
            out.append(key[0])

        return out


    adpositions = get_adpositions_multipartite(text)
    print ("Adpositions from the text: ",adpositions)

    # Get all the sentences for a given adpostion word. So each word may have mulitple sentences.
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt')
    from flashtext import KeywordProcessor

    def tokenize_sentences(text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        # Remove any short sentences less than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences
    sentences = tokenize_sentences(text)

    def get_sentences_for_keyword(keywords, sentences):
        keyword_processor = KeywordProcessor()
        keyword_sentences = {}
        for word in keywords:
            keyword_sentences[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                keyword_sentences[key].append(sentence)

        for key in keyword_sentences.keys():
            values = keyword_sentences[key]
            values = sorted(values, key=len, reverse=True)
            keyword_sentences[key] = values
        return keyword_sentences

    keyword_sentence_mapping_adpos = get_sentences_for_keyword(adpositions, sentences)

    for word in keyword_sentence_mapping_adpos:
        print (word, " : ",keyword_sentence_mapping_adpos[word],"\n")

    #  For every adposition word we have multiple sentences. For every sentence we blank the adposition word and ask BERT 
    #  to predict the top N choices. Then make a note of index of the correct answer in the predicitons. Then we sort the
    # sentences by the index and pick the top one.
    def get_best_sentence_and_options(word, sentences_array):
        keyword = word
        sentences = sentences_array
        sentences = sorted(sentences, key=len, reverse=False)
        max_no = min(5, len(sentences))
        sentences = sentences[:max_no]
        choices_filtered = []
        ordered_sentences = []
        for sentence in sentences:
            try:
                insensitive_line = re.compile(re.escape(keyword), re.IGNORECASE)
                no_of_replacements =  len(re.findall(re.escape(keyword),sentence,re.IGNORECASE))
                #blanked_sentence = sentence.replace(keyword, "____", 1)
                blanked_sentence = insensitive_line.sub("____", sentence)
                blanks = get_predicted_words(blanked_sentence)

                if blanks is not None:
                    choices_filtered = blanks
                    try:
                        word_index = choices_filtered.index(keyword.lower())
                        if no_of_replacements<2:
                            ordered_sentences.append((blanked_sentence, choices_filtered, word_index))
                    except:
                        pass
            except(ValueError):
                print("Here Error")
                    

        ordered_sentences = sorted(ordered_sentences, key=lambda x: x[2])
        if len(ordered_sentences) > 0:
            return (ordered_sentences[0][0], ordered_sentences[0][1])
        else:
            return None, None
    vocabmcqs = ""
    index = 1
    for each_adpos in adpositions:
        sentence, best_options = get_best_sentence_and_options(each_adpos, keyword_sentence_mapping_adpos[each_adpos])

        if sentence != None:
            print (sentence)
            vocabmcqs += str(index)+")   " +sentence + "\n\n"
            print (best_options)
            vocabmcqs += "options :-" + ", ".join(best_options)+"\n\n"
            print ("\n\n")
            index += 1
    return vocabmcqs
#full_text = "The Greek historian knew what he was talking about. The Nile River fed Egyptian civilization for hundreds of years. The Longest River the Nile is 4,160 miles long—the world’s longest river. It begins near the equator in Africa and flows north to the Mediterranean Sea. In the south the Nile churns with cataracts. A cataract is a waterfall. Near the sea the Nile branches into a delta. A delta is an area near a river’s mouth where the water deposits fine soil called silt. In the delta, the Nile divides into many streams. The river is called the upper Nile in the south and the lower Nile in the north. For centuries, heavy rains in Ethiopia caused the Nile to flood every summer. The floods deposited rich soil along the Nile’s shores. This soil was fertile, which means it was good for growing crops. Unlike the Tigris and Euphrates, the Nile River flooded at the same time every year, so farmers could predict when to plant their crops. Red Land, Black Land The ancient Egyptians lived in narrow bands of land on each side of the Nile. They called this region the black land because of the fertile soil that the floods deposited. The red land was the barren desert beyond the fertile region. Weather in Egypt was almost always the same. Eight months of the year were sunny and hot. The four months of winter were sunny but cooler. Most of the region received only an inch of rain a year. The parts of Egypt not near the Nile were a desert. Isolation The harsh desert acted as a barrier to keep out enemies. The Mediterranean coast was swampy and lacked good harbors. For these reasons, early Egyptians stayed close to home. Each year, Egyptian farmers watched for white birds called ibises, which flew up from the south. When the birds arrived, the annual flood waters would soon follow. After the waters drained away, farmers could plant seeds in the fertile soil. Agricultural Techniques By about 2400 B.C., farmers used technology to expand their farmland. Working together, they dug irrigation canals that carried river water to dry areas. Then they used a tool called a shaduf to spread the water across the fields. These innovative, or new, techniques gave them more farmland. Egyptian Crops Ancient Egyptians grew a large variety of foods. They were the first to grind wheat into flour and to mix the flour with yeast and water to make dough rise into bread. They grew vegetables such as lettuce, radishes, asparagus, and cucumbers. Fruits included dates, figs, grapes, and watermelons. Egyptians also grew the materials for their clothes. They were the first to weave fibers from flax plants into a fabric called linen. Lightweight linen cloth was perfect for hot Egyptian days. Men wore linen wraps around their waists. Women wore loose, sleeveless dresses. Egyptians also wove marsh grasses into sandals. Egyptian Houses Egyptians built houses using bricks made of mud from the Nile mixed with chopped straw. They placed narrow windows high in the walls to reduce bright sunlight. Egyptians often painted walls white to reflect the blazing heat. They wove sticks and palm trees to make roofs. Inside, woven reed mats covered the dirt floor. Most Egyptians slept on mats covered with linen sheets. Wealthy citizens enjoyed bed frames and cushions. Egyptian nobles had fancier homes with tree-lined courtyards for shade. Some had a pool filled with lotus blossoms and fish. Poorer Egyptians simply went to the roof to cool off after sunset. They often cooked, ate, and even slept outside. Egypt’s economy depended on farming. However, the natural resources of the area allowed other economic activities to develop too. The Egyptians wanted valuable metals that were not found in the black land. For example, they wanted copper to make tools and weapons. Egyptians looked for copper as early as 6000 B.C. Later they learned that iron was stronger, and they sought it as well. Ancient Egyptians also desired gold for its bright beauty. The Egyptian word for gold was nub. Nubia was the Egyptian name for the area of the upper Nile that had the richest gold mines in Africa. Mining minerals was difficult. Veins (long streaks) of copper, iron, and bronze were hidden inside desert mountains in the hot Sinai Peninsula, east of Egypt. Even during the cool season, chipping minerals out of the rock was miserable work. Egyptians mined precious stones too. They were probably the first people in the world to mine turquoise. The Egyptians also mined lapis lazuli. These beautiful blue stones were used in jewelry.The Nile had fish and other wildlife that Egyptians wanted. To go on the river, Egyptians made lightweight rafts by binding together reeds. They used everything from nets to harpoons to catch fish. One ancient painting even shows a man ready to hit a catfish with a wooden hammer. More adventurous hunters speared hippopotamuses and crocodiles along the Nile. Egyptians also captured quail with nets. They used boomerangs to knock down flying ducks and geese. (A boomerang is a curved stick that returns to the person who threw it.) Eventually, Egyptians equipped their reed boats with sails and oars. The Nile then became a highway. The river’s current was slow, so boaters used paddles to go faster when they traveled north with the current. Going south, they raised a sail and let the winds that blew in that direction push them. The Nile provided so well for Egyptians that sometimes they had surpluses, or more goods than they needed. They began to trade with each other. Ancient Egypt had no money, so people exchanged goods that they grew or made. This method of trade is called bartering. Egypt prospered along the Nile. This prosperity made life easier and provided greater opportunities for many Egyptians. When farmers produce food surpluses, the society’s economy begins to expand. Cities emerge as centers of culture and power, and people learn to do jobs that do not involve agriculture. For example, some ancient Egyptians learned to be scribes, people whose job was to write and keep records. As Egyptian civilization grew more complex, people took on jobs other than that of a farmer or scribe. Some skilled artisans erected stone or brick houses and temples. Other artisans made pottery, incense, mats, furniture, linen clothing, sandals, or jewelry. A few Egyptians traveled to the upper Nile to trade with other Africans. These traders took Egyptian products such as scrolls, linen, gold, and jewelry. They brought back exotic woods, animal skins, and live beasts. As Egypt grew, so did its need to organize. Egyptians created a government that divided the empire into 42 provinces. Many officials worked to keep the provinces running smoothly. Egypt also created an army to defend itself. One of the highest jobs in Egypt was to be a priest. Priests followed formal rituals and took care of the temples. Before entering a temple, a priest bathed and put on special linen garments and white sandals. Priests cleaned the sacred statues in temples, changed their clothes, and even fed them meals. Together, the priests and the ruler held ceremonies to please the gods. Egyptians believed that if the gods were angry, the Nile would not flood. As a result, crops would not grow, and people would die. So the ruler and the priests tried hard to keep the gods happy. By doing so, they hoped to maintain the social and political order. Slaves were at the bottom of society. In Egypt, people became slaves if they owed a debt, committed a crime, or were captured in war. Egyptian slaves were usually freed after a period of time. One exception was the slaves who had to work in the mines. Many died from the exhausting labor. Egypt was one of the best places in the ancient world to be a woman. Unlike other ancient African cultures, in Egyptian society men and women had fairly equal rights. For example, they could both own and manage their own property. The main job of most women was to care for their children and home, but some did other jobs too. Some women wove cloth. Others worked with their husbands in fields or workshops. Some women, such as Queen Tiy, even rose to important positions in the government. Children in Egypt played with toys such as "
#print(vocabexecute(full_text))