import math
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.sq_sum = 0  # 제곱합
        self.avg = 0
        self.std = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.sq_sum += (value ** 2) * n
        self.count += n
        self.avg = self.sum / self.count

        mean_sq = self.sq_sum / self.count
        var = mean_sq - self.avg ** 2
        if var < 0:
            var = 0.0  # numerical guard for tiny negative due to float error
        self.std = math.sqrt(var)  # 분산의 sqrt = 표준편차
