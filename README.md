# bolero-python-utils
각종 파이썬 언어 관련 유틸리티 스크립트(DL)  

## what is this?
딥러닝을 하면서 필요한 각종 유틸리티를 작성합니다.  

필요에 따라서는, 재사용이 가능한 함수도 제작합니다.  
(train_loss 기록, metric 구하는 함수 등)  

## How-to-use

### `~/.bashrc` command
```bash
$ git clone https://github.com/bolero2/bolero-python-utils.git
$ echo "export PYTHON_UTILS=$(pwd)/bolero-python-utils" >> ~/.bashrc && source ~/.bashrc
```

### Example Code for augmentation
```python
import os
import sys

PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
import _dl.augmentation as aug

print(aug.gaussian_blur)
print(aug.salt_and_pepper)
print(aug.gamma_correction)
```
