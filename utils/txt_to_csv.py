# Open text file 
file = open("smsspamcollection/SMSSpamCollection.txt", 'r', encoding = 'utf-8')

# Read file 
new_text = file.readlines()
	')
# Create a list to keep all the words in file 
words = []
line_break = 0

# Add all the words in file to list 
for x in range(0, len(new_text)):
	for word in new_text[x].split('\t'):
		words.append(word)



