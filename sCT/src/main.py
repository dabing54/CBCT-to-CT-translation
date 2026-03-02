import os
import yaml
from munch import DefaultMunch
from run.train import Trainer
from run.test import Tester
from run.fix_test import FixTester


def main():
    cfg = read_config()
    cfg = DefaultMunch.fromDict(cfg)  # dict转为对象
    if cfg.process == 1 or cfg.process == 2:
        trainer = Trainer(cfg)
        trainer.run()
    if cfg.process == 3:
        tester = Tester(cfg)
        tester.run()
    if cfg.process == 4:
        fix_tester = FixTester(cfg)
        fix_tester.run()
    # end if


def read_config():
    config_path = r'./config/base_config.yaml'  # 基础配置路径
    config = read_yaml(config_path)
    config_path = os.path.join(r'./config', config['config_file'])  # 额外配置路径
    config2 = read_yaml(config_path)
    config.update(config2)
    # 推断参数
    process = config['process']
    config['is_train'] = True if process == 1 or process == 2 else False
    config['is_continue_train'] = True if process == 2 else False
    config['is_test'] = True if process == 3 or process == 4 else False
    assert process in [1, 2, 3, 4]  # 新增内容需改前面几行
    return config


def read_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        result = yaml.safe_load(f)
    return result


if __name__ == '__main__':
    main()
