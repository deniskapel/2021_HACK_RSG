data:
	mkdir data
	mkdir logs
	wget https://russiansuperglue.com/tasks/download
	unzip download
	rm download
	rm -r __MACOSX
	mv -v combined data

models:
	# non-lemmatized
	wget http://vectors.nlpl.eu/repository/20/195.zip -P ./models
	# unzipping because it works faster when unzipped
	unzip ./models/195.zip -d ./models/wiki_rnc
	rm ./models/195.zip
	# lemmatized
	wget http://vectors.nlpl.eu/repository/20/199.zip -P ./models
	# unzipping because it works faster when unzipped
	unzip ./models/199.zip -d ./models/taiga
	rm ./models/199.zip

