# Build

```make``` - build docker image (genome_generator) and run (port 8080)

```make build``` - build docker image

```make run``` - run docker image (port 8080)

```make stop``` - stop docker image

# REST API

```GET /docs``` - swagger

![img.png](img.png)

```POST /generation/start?size={size}``` - start generation, returns task id

![img_1.png](img_1.png)

```GET /genome/status``` - returns status of all generation tasks

![img_2.png](img_2.png)

```GET /genome/{id}/status``` - return status of specific generation task

![img_3.png](img_3.png)

```GET /genome/download?filename={filename}``` - downloads generated file

![img_4.png](img_4.png)