from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_all_tasks():
    response = client.get("/genome/status")
    assert response.status_code == 200
    assert response.json() == {}


def test_start_generation():
    response = client.post("/generation/start?size=100")
    assert response.status_code == 200
    assert response.json()["taskId"] is not None


def test_get_task_status():
    generation_response = client.post("/generation/start?size=100")
    task_id = generation_response.json()["taskId"]
    task_response = client.get(f"/genome/{task_id}/status")
    assert task_response.status_code == 200
    task = task_response.json()
    assert task["id"] == task_id
    assert task["size"] == "100"


def test_download_file():
    file_response = client.get("/genome/download?filename=test.fasta")
    assert file_response.status_code == 200
