from Amazon_Reviews import get_data, load_data
# delete data_cache.pkl if you want to load a different file, otherwise it will load the cached data from the previous file

load_data() # Opens explorer to choose JSONL file with reviews and ratings



data = get_data() # Gets the data as the review text and rating only


 
for i in range(20):
        print("Item:",data[i])


