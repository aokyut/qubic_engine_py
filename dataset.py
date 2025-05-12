import sqlite3
from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from random import random
from tqdm import tqdm

to_bit_mask = 2 ** torch.arange(64)
hf_mask1 = 0x5555_5555_5555_5555
hf_mask2 = 0x3333_3333_3333_3333
vf_mask1 = 0x0f0f_0f0f_0f0f_0f0f
vf_mask2 = 0x00ff_00ff_00ff_00ff
df_mask1 = 0x0a0a_0a0a_0a0a_0a0a
df_mask2 = 0x00cc_00cc_00cc_00cc
dft_mask1 = 0x0505_0505_0505_0505
dft_mask2 = 0x0033_0033_0033_0033

def delta_swap(x: torch.Tensor, delta: int, mask: int):
    t = torch.bitwise_and(mask, (x.bitwise_xor(x.bitwise_right_shift(delta))))
    return x.bitwise_xor(t.bitwise_xor(t.bitwise_left_shift(delta)))

def horizontal_flip(x: torch.Tensor):
    x = delta_swap(x, 1, hf_mask1)
    x = delta_swap(x, 2, hf_mask2)
    return x

def vertical_flip(x: torch.Tensor):
    x = delta_swap(x, 4, vf_mask1)
    x = delta_swap(x, 8, vf_mask2)
    return x

def diagonal_flip(x: torch.Tensor):
    x = delta_swap(x, 3, df_mask1)
    x = delta_swap(x, 6, df_mask2)
    return x

def diagonal_flip_t(x: torch.Tensor):
    x = delta_swap(x, 5, dft_mask1)
    return delta_swap(x, 10, dft_mask2)

class BitRotation:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor):
        r = random()

        if r < 0.5:
            if r < 0.125:
                return diagonal_flip(horizontal_flip(x))
            elif r < 0.25:
                return diagonal_flip(x)
            elif r < 0.375:
                return horizontal_flip(diagonal_flip(x))
            else:
                return diagonal_flip_t(x)
        elif r < 0.75:
            if r < 0.625:
                return horizontal_flip(vertical_flip(x))
            else:
                return vertical_flip(x)
        elif r < 0.875:
            return x
        else:
            return horizontal_flip(x)



class QubicDataset(Dataset):
    def __init__(self, file_name, transforms=None, l=0.5, is_disc=False, label_smoothing=0.01):
        db = sqlite3.connect(file_name)
        _cur = db.cursor()

        _cur.execute("select att, def, flag, val from board_record")

        rows = _cur.fetchall()
        self.data = []
        for a, d, f, v in tqdm(rows):
            self.data.append((a, d, f, v))

        self.l = l
        self.transforms = transforms
        self.is_disc = is_disc
        self.label_smoothing = label_smoothing
    
    def __len__(self):
        # self.cur.execute("SELECT COUNT(*) FROM board_record")
        return len(self.data)

    def __getitem__(self, idx):
        # self.cur.execute(f"select att, def, flag, val from board_record where ROWID={idx + 1}")
        # a, d, f, v = self.cur.fetchone()
        a, d, f, v = self.data[idx]
        board = torch.tensor([a, d]).long()
        if self.transforms:
            board = self.transforms(board)
        board = board.unsqueeze(-1).bitwise_and(to_bit_mask).ne(0).float().reshape(-1)
        if self.is_disc:
            if f == 1:
                expect = torch.tensor([1.0, 0.0, 0.0]) * (1 - self.label_smoothing) \
                    + torch.tensor([0.0, 0.75, 0.25]) * self.label_smoothing
            elif f == -1:
                expect = torch.tensor([0.0, 0.0, 1.0]) * (1 - self.label_smoothing) \
                    + torch.tensor([0.25, 0.75, 0.0]) * self.label_smoothing
            else:
                expect = torch.tensor([0.0, 1.0, 0.0]) * (1 - self.label_smoothing) \
                    + torch.tensor([0.5, 0.0, 0.5]) * self.label_smoothing
        else:            
            f = f * 0.5 + 0.5
            expect = torch.tensor([f * self.l + (1 - self.l) * v]).float()

        return board, expect

class QubicDataModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, test_file, batch_size = 32, is_disc=False, label_smoothing=0.01):
        super().__init__()
        self.train = train_file
        self.valid = valid_file
        self.test = test_file
        self.batch_size = batch_size
        self.is_disc = is_disc
        self.ls = label_smoothing

    def setup(self, stage=None):
        self.train_dataset = QubicDataset(self.train, transforms=BitRotation(), is_disc=self.is_disc, label_smoothing=self.ls)
        self.valid_dataset = QubicDataset(self.valid, is_disc=self.is_disc, label_smoothing=self.ls)
        self.test_dataset = QubicDataset(self.test, is_disc=self.is_disc, label_smoothing=self.ls)
        # self.mnist_test = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self):
        pass