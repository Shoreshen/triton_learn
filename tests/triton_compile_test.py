import torch
import torch.nn as nn


class SiluModel(nn.Module):
    def __init__(self):
        super(SiluModel, self).__init__()
        self.act_fn = nn.functional.sigmoid

    def forward(self, x, up):
        x = self.act_fn(x) * x * up
        return x


def get_input(rank_id):
    input_ids = torch.rand((32, 1, 896), dtype=torch.float32)
    up = torch.rand((32, 1, 896), dtype=torch.float32)
    input_ids = input_ids.to("cuda:" + str(rank_id))
    up = up.to("cuda:" + str(rank_id))
    return input_ids, up


if __name__ == '__main__':
    model = SiluModel().to("cuda")
    model.eval()
    opt_model = torch.compile(model)
    input_ids, up = get_input(0)
    output = opt_model(input_ids, up)
    print("debug output:", output)