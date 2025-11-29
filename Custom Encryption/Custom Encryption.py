import random
import os


"""
=====================================================================================
=====================================================================================
FUNCTONS DEFINED
=====================================================================================
=====================================================================================
"""



def unique(listed):
    return_list = []
    for item in listed:
        if item not in return_list:
            return_list.append(item)
    return return_list



def convert_to_num(string, dictionary1, dictionary2, dictionary3, master_list):
    return_string = ""
    cap_string = string.upper()
    for text in cap_string:
        j = 0
        if master_list[j] % 3 == 1:
            return_string = return_string + dictionary1[text]
        if master_list[j] % 3 == 2:
            return_string = return_string + dictionary2[text]
        if master_list[j] % 3 == 0:
            return_string = return_string + dictionary3[text]
    return return_string



def convert_to_text(string, convert_dict1, convert_dict2, convert_dict3, master_list):
    new_string = ""
    for i in range(0,(int((len(string)-1)/2)+1)):
        j = 0
        if master_list[j] % 3 == 1:
            new_string = new_string + list(convert_dict1.keys())[list(convert_dict1.values()).index(string[2*i]+string[2*i + 1])]
        if master_list[j] % 3 == 2:
            new_string = new_string + list(convert_dict2.keys())[list(convert_dict2.values()).index(string[2*i]+string[2*i + 1])]
        if master_list[j] % 3 == 0:
            new_string = new_string + list(convert_dict3.keys())[list(convert_dict3.values()).index(string[2*i]+string[2*i + 1])]
        j += 1
    return new_string



def replacing(stringy):
    stringy = stringy.replace("AT", "@")
    stringy = stringy.replace("SH", "$")
    stringy = stringy.replace("OR", "|")
    return stringy



def dict_form():
    alphabet = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ",", ".",
            ";", ":", "{", "}", "[", "]", "\\", "|", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "_", "!", "@", "#", "$", "%", "^", "&", "*", "(",
            ")", "+", "=", "?", "'", "…", "’"]
    numbers = ["00","01", "02", "03", "04", "05", "06", "07", "08", "09"]
    for i in range(10, len(alphabet) + 400):
        new_num = str(random.randint(10,99))
        numbers.append(new_num)
    x = unique(numbers)
    random.shuffle(x)
    convert_dict1 = dict(zip(alphabet, x))
    random.shuffle(x)
    convert_dict2 = dict(zip(alphabet, x))
    random.shuffle(x)
    convert_dict3 = dict(zip(alphabet, x))
    return convert_dict1, convert_dict2, convert_dict3 



"""
========================================================================================
========================================================================================
MAIN FUNCTION
========================================================================================
========================================================================================
"""
def main():
    seed = input("Please enter your seed.\n> ")
    os.system('cls')
    random.seed(int(seed))
    convert_dict1, convert_dict2, convert_dict3 = dict_form()
    decision = input("Decrypt or Encrypt? (Please enter 'decrypt' or 'encrypt')\n> ")
    if decision.upper() in ["DECRYPT", "D", "1"]:
        os.system('cls')
        master_list = []
        stringy = input("Please enter the encrypted text\n> ")
        os.system('cls')
        for num in range(1,500001):
            master_list.append(random.randint(1,10))
        converted = convert_to_text(stringy, convert_dict1, convert_dict2, convert_dict3, master_list)
        print("Your Message: \n\n" + converted + "\n\n")
        
    if decision.upper() in ["ENCRYPT", "E", "2"]:
        os.system('cls')
        master_list = []
        stringy = input("What is your message? > ")
        os.system('cls')
        for num in range(1,500001):
            master_list.append(random.randint(1,10))
        temp = convert_to_num(replacing(stringy.upper()), convert_dict1, convert_dict2, convert_dict3, master_list)

        #converted = convert_to_num(temp, convert_dict)
        print("Your Encrypted Message: \n\n" + temp + "\n\n")
    
main()
