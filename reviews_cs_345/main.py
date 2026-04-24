from amazon_reviews import get_data, load_data


load_data() # Opens explorer to choose JSONL file with reviews and ratings
data = get_data()

for i in range(20):
        print("Item:",data[i])


