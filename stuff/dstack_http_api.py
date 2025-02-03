import os
from pathlib import Path
import time
import requests

url = os.environ["DSTACK_URL"]
token = os.environ["DSTACK_TOKEN"]
project = os.environ["DSTACK_PROJECT"]
ssh_public_key = Path(os.environ["SSH_PUBLIC_KEY_PATH"]).read_text()

print("Initializing repo")
resp = requests.post(
    url=f"{url}/api/project/{project}/repos/init",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "repo_id": "my_dstack_private_repo",
        "repo_info": {
            "repo_type": "remote",
            "repo_name": "my_dstack_private",
        },
        "repo_creds": {
            "protocol": "https",
            "clone_url": "https://github.com/r4victor/my_dstack_private.git",
            "oauth_token": os.environ["GITHUB_TOKEN"],
        }
    },
)

print("Submitting task")
resp = requests.post(
    url=f"{url}/api/project/{project}/runs/apply",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "plan":{
            "run_spec": {
                "configuration": {
                    "type": "task",
                    "commands": [
                        "echo Start",
                        "ls -a /workflow", # do some work here
                        "echo Finish"
                    ],
                },
                "ssh_key_pub": ssh_public_key,
                "repo_id": "my_dstack_private_repo",
                "repo_data": {
                    "repo_type": "remote",
                    "repo_name": "my_dstack_private",
                    "repo_branch": "master",
                }
            }
        },
        "force": False,
    },
)
print(resp.json())
run_name = resp.json()["run_spec"]["run_name"]

print("Waiting for task completion")
while True:
    resp = requests.post(
        url=f"{url}/api/project/{project}/runs/get",
        headers={"Authorization": f"Bearer {token}"},
        json={"run_name": run_name}
    )
    if resp.json()["status"] in ["terminated", "aborted", "failed", "done"]:
        print(f"Run finished with status {resp.json()['status']}")
        break
    time.sleep(2)
