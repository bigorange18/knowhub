## pytest

### 配置环境

```bash
conda create -n py11 python=3.11.11
pip install -r requirement.txt
```



只是用第0,1张显卡

```python
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
```





