from dstack.api import Client, Task
from dstack.api._public.resources import GPU, Resources

task = Task(
    image="ghcr.io/huggingface/text-generation-inference:latest",
    env={"MODEL_ID": "TheBloke/Llama-2-13B-chat-GPTQ"},
    commands=[
        "text-generation-launcher --trust-remote-code --quantize gptq",
    ],
    ports=["80"],
    resources=Resources(gpu=GPU(memory="24GB")),
)

client = Client.from_config()
run = client.runs.submit(
    run_name="my-awesome-run",
    configuration=task,
    repo=None,
)
