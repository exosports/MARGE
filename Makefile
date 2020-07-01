
bart:
	@if [ ! -d "./modules/BART" ]; then                                       \
		echo "\nCloning BART...";                                             \
		git clone --recursive https://github.com/exosports/BART modules/BART/;\
		echo "Finished cloning BART into 'modules'.\n";                       \
	else                                                                      \
		echo "BART already exists.\n";                                        \
	fi
	@echo "\nCompiling BART..."
	@cd ./modules/BART/modules/transit && make
	@cd ./modules/BART/modules/MCcubed && make
	@echo "Finished compiling BART.\n"

TLI:
	./modules/BART/modules/transit/pylineread/src/pylineread.py -c $(cfile)
