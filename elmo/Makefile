data:
	mkdir data
	mkdir logs
	mkdir submissions
	wget https://russiansuperglue.com/tasks/download
	unzip download
	rm download
	rm -r __MACOSX
	mv -v combined data

models:
	# test third-party models
	# tokenized
	wget http://vectors.nlpl.eu/repository/20/195.zip -P ./models
	# unzipping because it works faster when unzipped
	unzip ./models/195.zip -d ./models/wiki_rnc
	rm ./models/195.zip
	# lemmatized
	wget http://vectors.nlpl.eu/repository/20/199.zip -P ./models
	# unzipping because it works faster when unzipped
	unzip ./models/199.zip -d ./models/taiga
	rm ./models/199.zip
	# ud model
	wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/russian-taiga-ud-2.5-191206.udpipe -P ./models/ud

