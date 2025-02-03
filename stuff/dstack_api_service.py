import os
from dstack.api.server import APIClient

url = os.environ["DSTACK_URL"]
token = os.environ["DSTACK_TOKEN"]
project = os.environ["DSTACK_PROJECT"]

client = APIClient(base_url=url, token=token)

run = client.runs.get(project, "my-run")
new_run_spec = run.run_spec
new_run_spec.configuration.replicas = 3

plan = client.runs.get_plan(project, new_run_spec)
updated_run = client.runs.apply_plan(project, plan)
