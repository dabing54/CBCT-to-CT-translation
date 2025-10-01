import torch


def lr_scheduler_multi_step(optimizer, warmup_num, milestone_list, gamma):
    """自定义学习率调整方案，预热后按MultiStepLR方式进行衰减"""
    def func(step):
        if step < warmup_num:
            frac = (step + 0.5) / warmup_num
        else:
            frac = 1
            max_item = len(milestone_list)
            i = 0
            while i < max_item:
                if step > milestone_list[i]:
                    frac *= gamma
                    i += 1
                else:
                    i = max_item
        return frac

    return torch.optim.lr_scheduler.LambdaLR(optimizer, func)


def lr_scheduler_poly(optimizer, warmup_num, max_epoch, gamma):
    """自定义学习率调整方案，预热后按Poly方式进行衰减"""
    def func(step):
        if step < warmup_num:
            frac = (step + 0.5) / warmup_num
        elif step < max_epoch:
            frac = (1 - (step - warmup_num) / max_epoch) ** gamma
        else:
            frac = (1 / max_epoch) ** gamma
        return frac

    return torch.optim.lr_scheduler.LambdaLR(optimizer, func)
