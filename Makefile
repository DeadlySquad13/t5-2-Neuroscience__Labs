install: requirements.txt
	pip install -r requirements.txt

install-dev: requirements.txt requirements_dev.txt 
	make install
	pip install -r requirements_dev.txt
	pip install -e .

install-dev-ju: requirements.txt requirements_dev.txt requirements_dev_ju.txt
	make install-dev
	pip install -r requirements_dev_ju.txt

install-dev-ju-nvim: requirements.txt requirements_dev.txt requirements_dev_ju.txt requirements_dev_ju_nvim.txt
	make install-dev-ju
	pip install -r requirements_dev_ju_nvim.txt

# Lidar specific (lab6).
install-lidar: requirements_lidar.txt
	pip install -r requirements_lidar.txt

regenerate_requirements: src
	pigar generate src
