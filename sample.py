from model import NNEvaluator, DiscEvaluator
from dataset import QubicDataset
import torch
import math



def cel(t, x):
    return - t * math.log(x) - (1 - t) * math.log(1 - x)

def tce(x1, x2, x3, t1, t2, t3):
    return - t1 * (math.log(x1) - math.log(t1)) - t2 * (math.log(x2) -math.log(t2)) - t3 * (math.log(x3) - math.log(t3))


def model_check(model):
    loss = 0
    size = len(ds)
    for j in range(size):
        i, t = ds[j]
        cel_loss = cel(t.item(), model(i).item())
        # print(cel_loss)
        loss += cel_loss
        # print(t, model(i))
    print(loss / size)

def loss_check():
    loss = 0
    size = len(ds)
    for j in range(size):
        i, t = ds[j]
        cel_loss = cel(t, t)
        loss += cel_loss
    print(loss / size)

ds = QubicDataset("./valid_date_depth5_m.db")

# loss_check()
# exit(0)

model = NNEvaluator()

# model_check(model)

model = model.load_from_checkpoint("/Users/aokiyuuta/project_python/qubic_engine_py/pretrained/nneval-val_loss=0.039.ckpt", strict=True)
model.eval()

model.to_onnx("nneval.onnx", torch.zeros(128))
# model_check(model)
