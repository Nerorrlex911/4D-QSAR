from torch.utils.data import DataLoader,Dataset

class MolDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()