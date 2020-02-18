
bart:
	@if [ ! -d "./modules/BART" ]; then                                       \
		echo "\nCloning BART..."                                              \
		git clone --recursive https://github.com/exosports/BART modules/BART/;\
		echo "Finished cloning BART into 'modules'.\n";                       \
		echo "Switching to the compatible BART version...";                   \
		cd modules/BART;                                                      \
		git checkout e04f29f95833203d32c78bab7f688132b393009c;                \
		cd ../..;                                                             \
	else                                                                      \
		echo "BART already exists.\n";                                        \
	fi
	@echo "\nModifying files within BART and MCcubed..."
	@yes | cp -R code/BART/ modules/
	@yes | cp -R code/transit/ modules/BART/modules/
	@yes | cp -R code/MCcubed/ modules/BART/modules/
	@echo "\nCompiling BART..."
	@cd ./modules/BART/modules/transit && make
	@cd ./modules/BART/modules/MCcubed && make
	@echo "Finished compiling BART.\n"

TLI:
	./modules/BART/modules/transit/pylineread/src/pylineread.py -c $(cfile)
