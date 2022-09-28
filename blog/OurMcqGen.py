

def excecute(full_text,compression_ratio=0.8,summarizeFeature=False):

    from summarizer import Summarizer
    import os

    #f = open("sun.txt","r")
    #full_text = f.read()
    #full_text = "Scientists know many things about the Sun. They know how old it is. The Sun is more than 4 billion years old. That would be too many candles to put on a birthday cake!  They also know the Sun's size. The Sun may seem small, but that is because it is so far away. It is about 93 million miles (150 million kilometers) away from the Earth. The Sun is so large that the diameter of the Sun is 109 times the Earth's diameter. The Sun also weighs as much as 333,000 Earths. The Sun is made up of gases: 75% hydrogen and 25% helium.  Hydrogen is the simplest and lightest of all of the known elements. When you combine hydrogen with oxygen, you get water. You probably know what helium is. It is the gas that can be put into balloons to make them stay in the air and float. Scientists also know the temperature of the Sun. The surface of the Sun is about 10,000 degrees Fahrenheit (5,600 degrees Celsius). That might sound hot, but the Sun's core is even hotter. The core is the central region where the temperature reaches about 27 million degrees Fahrenheit (15 million Celsius). The Sun is the center of our Solar System. Besides the Sun, the Solar System is made up of the planets, moons, asteroid belt, comets, meteors, and other objects. The Earth and other planets revolve around the Sun. The Sun is very important. Without it, there would be only darkness and our planet would be very cold and be without liquid water. Our planet would also be without people, animals, and plants because these things need sunlight and water to live. The Sun also gives out dangerous ultraviolet light which causes sunburn and may cause cancer. That is why you need to be careful of the Sun and wear sunscreen and clothing to protect yourself from its rays. Scientists have learned many things about the Sun. They study the Sun using special tools or instruments such as telescopes. One thing they do is to look at the amount of light from the Sun and the effect of the Sun's light on the Earth's climate. The Sun is actually a star. It is the closest star to the Earth.  Scientists also study other stars, huge balls of glowing gas in the sky. There are over 200 billion stars in the sky. Some are much larger than the Sun and others are smaller than the Earth. They all look tiny because they are so far away from the Earth. This distance is measured in light-years, not in miles or kilometers. One light-year is equal to the distance that light travels in one year. This is about six trillion miles or ten trillion kilometers!  Stars look like they are twinkling because when we see them, we are looking at them through thick layers of turbulent (moving) air in the Earth's atmosphere. That is why the words are written in the song: Twinkle, Twinkle, Little Star. Stars have lifetimes of billions of years. They are held together by their own gravity. Over half of the stars in the sky are in groups of two. They orbit around the same center point and across from each other. There are also larger groups of stars called clusters. These clusters of stars make up galaxies. Our Solar System is located in the Milky Way Galaxy. "

    model = Summarizer()
    result = model(full_text, min_length=20, max_length = 1000 , ratio = compression_ratio)

    summarized_text = ''.join(result)
    print (summarized_text)
    if summarizeFeature==True:
        return summarized_text


    import pprint
    import itertools
    import re
    import pke
    import string
    from nltk.corpus import stopwords

    def get_nouns_multipartite(text):
        out=[]

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'PROPN','NOUN'}
        #pos = {'VERB', 'ADJ', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=0.5,
                                    threshold=0.0,
                                    method='average')
        keyphrases = extractor.get_n_best(n=30)

        for key in keyphrases:
            out.append(key[0])

        return out

    keywords = get_nouns_multipartite(full_text) 
    print (keywords)
    filtered_keys=[]
    for keyword in keywords:
        if keyword.lower() in summarized_text.lower():
            filtered_keys.append(keyword)
            
    print (filtered_keys)
    #filtered_keys = keywords

    from nltk.tokenize import sent_tokenize
    from flashtext import KeywordProcessor

    def tokenize_sentences(text):
        sentences = [sent_tokenize(text)]
        sentences = [y for x in sentences for y in x]
        # Remove any short sentences less than 20 letters.
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

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

    sentences = tokenize_sentences(summarized_text)
    keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)
        
    print (keyword_sentence_mapping)

    import requests
    import json
    import re
    import random
    from pywsd.similarity import max_similarity
    from pywsd.lesk import adapted_lesk
    from pywsd.lesk import simple_lesk
    from pywsd.lesk import cosine_lesk
    from nltk.corpus import wordnet as wn

    # Distractors from Wordnet
    def get_distractors_wordnet(syn,word):
        distractors=[]
        word= word.lower()
        orig_word = word
        if len(word.split())>0:
            word = word.replace(" ","_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0: 
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            #print ("name ",name, " word",orig_word)
            if name == orig_word:
                continue
            name = name.replace("_"," ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
        return distractors

    def get_wordsense(sent,word):
        word= word.lower()
        
        if len(word.split())>0:
            word = word.replace(" ","_")
        
        
        synsets = wn.synsets(word,'n')
        if synsets:
            wup = max_similarity(sent, word, 'wup', pos='n')
            adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
            lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    # Distractors from http://conceptnet.io/
    def get_distractors_conceptnet(word):
        word = word.lower()
        original_word= word
        if (len(word.split())>0):
            word = word.replace(" ","_")
        distractor_list = [] 
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
        obj = requests.get(url).json()

        for edge in obj['edges']:
            link = edge['end']['term']

            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
            obj2 = requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
                    
        return distractor_list

    key_distractor_list = {}

    for keyword in keyword_sentence_mapping:
        try:
            if len(keyword_sentence_mapping[keyword]) == 0:
                continue
            wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
            if wordsense:
                print(wordsense)
                distractors = get_distractors_wordnet(wordsense,keyword)
                if len(distractors) ==0:
                    distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
            else:
                
                distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
        except:
            print("Here Error")

    index = 1
    mcqs = ""
    print ("#############################################################################")
    print ("NOTE::::::::  Since the algorithm might have errors along the way, wrong answer choices generated might not be correct for some questions. ")
    print ("#############################################################################\n\n")
    for each in key_distractor_list:
        sentence = keyword_sentence_mapping[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub( " _______ ", sentence)
        mcqs += "%s)"%(index)+" "+output+"\n"     #up
        print ("%s)"%(index),output)
        choices = [each.capitalize()] + key_distractor_list[each]
        top4choices = choices[:4]
        random.shuffle(top4choices)
        optionchoices = ['a','b','c','d']
        for idx,choice in enumerate(top4choices):
            print ("\t",optionchoices[idx],")"," ",choice)
            mcqs += "\t"+optionchoices[idx]+")"+" "+choice
        print ("\n\tMore options: ", choices[4:9],"\n\n")
        mcqchoices = ", ".join(choices[4:9])
        mcqs += "\n\tMore options: "+ mcqchoices +"\n\n"
        index = index + 1
    return mcqs

#full_text = "Scientists know many things about the Sun. They know how old it is. The Sun is more than 4 billion years old. That would be too many candles to put on a birthday cake!  They also know the Sun's size. The Sun may seem small, but that is because it is so far away. It is about 93 million miles (150 million kilometers) away from the Earth. The Sun is so large that the diameter of the Sun is 109 times the Earth's diameter. The Sun also weighs as much as 333,000 Earths. The Sun is made up of gases: 75% hydrogen and 25% helium.  Hydrogen is the simplest and lightest of all of the known elements. When you combine hydrogen with oxygen, you get water. You probably know what helium is. It is the gas that can be put into balloons to make them stay in the air and float. Scientists also know the temperature of the Sun. The surface of the Sun is about 10,000 degrees Fahrenheit (5,600 degrees Celsius). That might sound hot, but the Sun's core is even hotter. The core is the central region where the temperature reaches about 27 million degrees Fahrenheit (15 million Celsius). The Sun is the center of our Solar System. Besides the Sun, the Solar System is made up of the planets, moons, asteroid belt, comets, meteors, and other objects. The Earth and other planets revolve around the Sun. The Sun is very important. Without it, there would be only darkness and our planet would be very cold and be without liquid water. Our planet would also be without people, animals, and plants because these things need sunlight and water to live. The Sun also gives out dangerous ultraviolet light which causes sunburn and may cause cancer. That is why you need to be careful of the Sun and wear sunscreen and clothing to protect yourself from its rays. Scientists have learned many things about the Sun. They study the Sun using special tools or instruments such as telescopes. One thing they do is to look at the amount of light from the Sun and the effect of the Sun's light on the Earth's climate. The Sun is actually a star. It is the closest star to the Earth.  Scientists also study other stars, huge balls of glowing gas in the sky. There are over 200 billion stars in the sky. Some are much larger than the Sun and others are smaller than the Earth. They all look tiny because they are so far away from the Earth. This distance is measured in light-years, not in miles or kilometers. One light-year is equal to the distance that light travels in one year. This is about six trillion miles or ten trillion kilometers!  Stars look like they are twinkling because when we see them, we are looking at them through thick layers of turbulent (moving) air in the Earth's atmosphere. That is why the words are written in the song: Twinkle, Twinkle, Little Star. Stars have lifetimes of billions of years. They are held together by their own gravity. Over half of the stars in the sky are in groups of two. They orbit around the same center point and across from each other. There are also larger groups of stars called clusters. These clusters of stars make up galaxies. Our Solar System is located in the Milky Way Galaxy. "
#excecute(full_text)