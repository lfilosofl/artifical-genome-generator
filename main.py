import asyncio
import logging
import uuid

from genome_generator.app import TrainApp
from genome_generator.generator import GenomeGenerator
from genome_generator.logging_config import init_logging
from fastapi import FastAPI
from datetime import datetime
from fastapi.responses import FileResponse

app = FastAPI()
generator = GenomeGenerator('data/models/genome_generator.keras')
queue = asyncio.Queue()
tasks = {}

logger = logging.getLogger(__name__)


class Task:
    pass


async def generate_genome_from_queue(queue):
    while True:
        task = await queue.get()
        filename = "data/output/generated_genome_" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".fasta"
        generator.generate_genomes_file(int(task.size), filename)
        queue.task_done()
        task.filename = filename
        task.status = "Finished"


@app.post("/generation/start")
async def generate(size):
    task = Task()
    task.size = size
    task.id = str(uuid.uuid4())
    await queue.put(task)
    tasks[task.id] = task
    return {"taskId": task.id}


@app.get("/genome/status")
async def get_all_tasks_generation_status():
    return tasks


@app.get("/genome/{id}/status")
async def get_task_generation_status(id):
    if id in tasks:
        return tasks[id]
    return "Unknown task id"


@app.get("/genome/download")
async def download(filename):
    return FileResponse(path="data/output/" + filename, filename=filename, media_type='text/plain')


async def start_training(train_app):
    train_app.train()


async def main():
    # todo worker should be in separate process to avoid blocking main event loop
    # todo replace asyncio queue with smth persistent/distributed (e.g. celery)??
    task_worker = asyncio.create_task(generate_genome_from_queue(queue))

    train_app = TrainApp()
    asyncio.create_task(start_training(train_app))


@app.on_event("startup")
async def startup_event():
    init_logging()
    await main()
