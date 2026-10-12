# EXP-TRAIN-003：诊断后确定的方法训练

状态：planned / gated，不得现在运行。

只有 EXP-DIAG-003 与 EXP-ABL-001 完成后，才填写方法 ID、唯一自变量和配置。该实验用于检验 selector/其他被证实组件能否在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下超越干净 baseline。

若诊断表明 selection 不是主要瓶颈，本实验可以取消或改名，但 ID 不复用。
