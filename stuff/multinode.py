import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.distributed.elastic.multiprocessing.errors import record

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


class MyTrainDataset(Dataset):
    """Custom Dataset for training data."""

    def __init__(self, size):
        """
        Initialize the dataset with random data.

        Args:
            size (int): The size of the dataset.
        """
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, index):
        """
        Get an item from the dataset at a given index.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: A tuple containing the input data and target.
        """
        return self.data[index]


def ddp_setup():
    """Set up the distributed data parallel (DDP) environment."""
    init_process_group(backend="nccl",  init_method='env://', world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))
    print("LOCAL RANK: ", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    #os.environ["HIP_VISIBLE_DEVICES"] = int(os.environ["LOCAL_RANK"])
    #os.environ["UCX_NET_DEVICES"] = "bnxt_re3:1"
    #os.environ["NCCL_SOCKET_IFNAME"] = "enp139s0np0"

class Trainer:
    """Trainer class to handle training loop, snapshots, and DDP."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            train_data (DataLoader): The DataLoader for training data.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            save_every (int): How often to save snapshots.
            snapshot_path (str): Path to save the snapshots.
        """
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        """
        Load a training snapshot to resume training.

        Args:
            snapshot_path (str): Path to the snapshot file.
        """
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        """
        Run a single batch through the model.

        Args:
            source (torch.Tensor): Input data.
            targets (torch.Tensor): Target data.
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        """
        Run a single epoch of training.

        Args:
            epoch (int): The current epoch number.
        """
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        """
        Save a snapshot of the model and training state.

        Args:
            epoch (int): The current epoch number.
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        """
        Run the training loop for a given number of epochs.

        Args:
            max_epochs (int): The total number of epochs to train.
        """
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    """
    Load the training objects: dataset, model, and optimizer.

    Returns:
        tuple: A tuple containing the dataset, model, and optimizer.
    """
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Prepare the DataLoader for the dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: The prepared DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


@record
def main(
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
):
    """
    Main function to set up DDP, load data, and start training.

    Args:
        save_every (int): How often to save snapshots.
        total_epochs (int): The total number of epochs to train.
        batch_size (int): The batch size for training.
        snapshot_path (str, optional): Path to save snapshots. Defaults to "snapshot.pt".
    """
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    print ('Local rank: ', LOCAL_RANK)
    print ('World rank: ', WORLD_RANK)
    print ('World size: ', WORLD_SIZE)

    parser = argparse.ArgumentParser(description="Simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)
