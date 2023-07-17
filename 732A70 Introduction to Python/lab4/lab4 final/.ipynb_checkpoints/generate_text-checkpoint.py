import sys
import random
from text_stats import compute_all_sucessors 
import time


if __name__ == '__main__':  #This is to let the script directly callable

    start_time = time.time() 
    
    if len(sys.argv) < 4:
        print("Input is wrong, Please use: python generate_text.py <filename> <starting_word> <max_words>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        with open(filename, encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print("The file does not exist!")
        sys.exit(1)

    start_word = sys.argv[2].lower()
    max_words = int(sys.argv[3])   
    msg = [start_word]
    current_word = start_word
    successors_dic = compute_all_sucessors(text)
    
    while len(msg) < max_words:
        #print("Processing pls wait")
        
        if start_word not in successors_dic:
            print("-----WARNING-----")
            print("The starting_word is not in the text")
            print("----------------\n")
            break
        if len(successors_dic[current_word]) == 0:
            break

        choices, weights = zip(*successors_dic[current_word].items())
        probs = [count/sum(weights) for count in weights]
        next_word = random.choices(choices, weights=probs)[0]
        msg.append(next_word) 
        current_word = next_word 

       
            
    #print(msg)
    print("-----MSG-----")
    print(' '.join(msg)) 
    print("----------------\n")
    print(f"total words: {len(msg)}")
    
    end_time = time.time() # record end time
    run_time = end_time - start_time # calculate run time
    print(f"Time used: {run_time}")
    
    
    if len(sys.argv) == 5:
        output_filename = sys.argv[4]
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            
            output_file.write("----------------\n")
            output_file.write(f"{' '.join(msg)}\n")
            output_file.write("----------------\n")
            