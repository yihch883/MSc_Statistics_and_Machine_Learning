import re
from collections import Counter
import sys
import time

def get_every_word(text):
    """
    function takes input text and split into words and converts to lower case
    """
    return  re.findall(r"\b\w+\b", text.lower()) 


def get_word_count_dic(text):
    """
    takes input text,splits into words,converts to lowercase and counter the number of occurences of the word
    """
    # the syntax is to find all the words, with \b is use to define beginning or end of a word, \w is to match Unicode word characters
    words = re.findall(r"\b\w+\b", text.lower())
    return Counter(words)

def get_letter_count_dic(text):
    """
    takes input text and return a dictionary of counts of alphabets occurences
    """
    return Counter(character for character in text.lower() if character.isalpha())

def get_one_word_successors(current_word, text): 
    # we don't actually use this function, since we decide to calculate all words and their sucessor once
    # However, we keep the function here since it can be use to compute successor on the fly         
    following_words = Counter(re.findall(rf"\b{current_word.lower()}\s+(\w+)\b", text.lower()))
    return following_words

def compute_all_sucessors(text):
    """
    takes input text and computes successors to each word
    """
   
    words = re.findall(r"\b\w+\b", text.lower())
    last_word = words[-1]
    successors_dic = {}
    for i in range(len(words) - 1):
        word = words[i]
        if word not in successors_dic:
            successors_dic[word] = {}
        next_word = words[i+1]
        if next_word not in successors_dic[word]:
            successors_dic[word][next_word] = 0
        successors_dic[word][next_word] += 1
        
    if last_word not in successors_dic:# This is to make sure the last word is in the sucessors_dic as we only loop po n-1 word
        successors_dic[last_word] = {}
    return successors_dic


if __name__ == '__main__':  #This is to let the script directly callable
    start_time = time.time() 
    if len(sys.argv) < 2:
        print("Input is wrong, Please use: python text_stats.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("The file does not exist!")
        sys.exit(1)

    word_count_dic = get_word_count_dic(text)
    letter_count_dic = get_letter_count_dic(text)
    successors_dic = compute_all_sucessors(text)
    
    # print number of words
    num_words = sum(word_count_dic.values())
    print("----------------\n")
    print(f"Number of words: {num_words}")

    # print number of unique words
    num_unique_words = len(word_count_dic)
    print(f"Number of unique words: {num_unique_words}")
    print("----------------\n")
    
    # print frequency table for alphabetic letters
    print("Letter frequency:")
    for letter, count in letter_count_dic.most_common(): #most_common is use to sort the Counter object in to list
        print(f"{letter}: {count}")

    # print 5 most common words and 3 words that follow them
    common_words = word_count_dic.most_common(5) 
    print("----------------\n")
    print("Most common words:")
  
    
    for word, count in common_words:
        print(f"{word} ({count} times)")
        
        following_words = Counter(successors_dic[word]).most_common(3)
        
        for following_word, following_count in following_words:
            print(f"- {following_word}, {following_count}")

    
    end_time = time.time() # record end time
    run_time = end_time - start_time # calculate run time
    print(f"Time used: {run_time}")
    
    if len(sys.argv) == 3:
        output_filename = sys.argv[2]
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            
            output_file.write("----------------\n")
            output_file.write(f"Number of words: {num_words}\n")
            output_file.write(f"Number of unique words: {num_unique_words}\n")
            output_file.write("----------------\n")
            output_file.write("Letter frequency:\n")
            for letter, count in letter_count_dic.most_common():
                output_file.write(f"{letter}: {count}\n")
            output_file.write("----------------\n")
            output_file.write("Most common words:\n")
            for word, count in common_words:
                output_file.write(f"{word} ({count} times)\n")
                following_words = Counter(successors_dic[word]).most_common(3)
                for following_word, following_count in following_words:
                    output_file.write(f"- {following_word}, {following_count}\n")
                    
        