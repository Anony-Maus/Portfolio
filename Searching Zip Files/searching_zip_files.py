import os
import zipfile
import time


start = time.time()
os.chdir(os.path.dirname(__file__))

## Fill in with the name of the file. Must be case accurate!
file_name = ["file.zip"]


## Fill in below with the search text.
search_string = "99999999"
#################################################################################################



for files in file_name:
    search_string = search_string.lower()
    search_bytes = bytes(search_string, encoding = "utf-8")
    file_ref = files.rstrip(".zip") + "/"
    #print(file_ref)

    with zipfile.ZipFile(files, 'r') as zip_ref:
        listed = zip_ref.namelist()
        
    new_list = [file for file in listed if file.endswith(".txt") or file.endswith(".csv")]
    #print(new_list)
    newer_list = [string for string in new_list if file_ref not in string]
    #print(newer_list)
    newest_list = [string.strip() for string in newer_list]
    #print(newest_list)
    os.chdir(os.getcwd())

    hit_files = []


    for file in newest_list:
        with zipfile.ZipFile(files, 'r') as zippy:
            text = zippy.read(file)
            text = text.lower()
            if search_bytes in text:
                hit_files.append(file)
        
    # Outputs the files that contain the given string.
    print("\n\n"+ files +"\n\nNumber of files with the given text: {}\n\n" .format(len(hit_files)))
    print("Here is the list of files containing the given text. \n" + str(hit_files))


end = time.time()

time_taken = end - start

print("\n\nThis process took {:.2f} seconds.".format(time_taken))
