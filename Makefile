all: build run
build:
	docker build -t genome_generator .
run:
	docker run -d -p 8080:80 genome_generator
stop:
	docker stop `(docker ps -a -q  --filter ancestor=genome_generator)`
