import torch


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name="embedding"):
        # emb_name 这个参数要换成模型中 embedding 的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="embedding"):
        # emb_name 这个参数要换成模型中 embedding 的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


'''
使用例子：
fgm = FGM(model, epsilon=1, emb_name="embedding") # (#1) 初始化

for batch_input, batch_label in data:
    loss = model(batch_input, batch_label) # 正常训练
    loss.backward() # 反向传播，得到正常的 grad
    
    # 对抗训练
    fgm.attack() #（#2）在 embedding 上添加对抗扰动
    loss_adv = model(batch_input, batch_label) # (#3) 计算含有扰动的对抗样本的 loss
    loss_adv.backward() # (#4) 反向传播，并在正常的 grad 基础上，累加对抗训练的梯度
    fgm.restore() # (#5) 恢复 embedding 参数
    
    # 梯度下降，更新参数
    optimizer.step()
    model.zero_grad()
'''
