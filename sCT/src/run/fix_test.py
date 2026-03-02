from run.run_base import RunBase
from run.fix_evaluate import FixEvaluator


class FixTester(RunBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fix_evaluator = FixEvaluator(cfg, self.data_loader_builder, self.model)

    def run(self):
        self.fix_evaluator.run(show_plot=True)
