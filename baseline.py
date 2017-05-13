"""
Gathers filler words
"""

DATADIR = "./swb_ms98_transcriptions/"

# Ideally I would want something like a disfluency annotated data alongside
# the word-timed data but unfortunately I only have a sample of disfluency data
FILLERS = ['um', 'uh', 'you know', 'like', 'huh', 'uh-huh', 'um-hum']

def main():
	for dirpath, dirnames, filenames in os.walk(DATADIR):

if __name__ == '__main__':
	main()