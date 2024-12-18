## 小技巧

### 1、添加代理下载git代码仓

```shell
git clone https://ghp.ci/你需要下载的代码仓
```

eg:

```shell
git clone https://ghp.ci/https://github.com/bigorange18/knowhub.git
```


### 2、使用token提交代码

1.已经有仓库:为仓库添加token

```shell

git remote set-url origin https://<你的token>@github.com/<GitHub用户名>/<要提交到的库名>.git
```


```shell
2.原本没有仓库: git clone 添加token

git clone https://<你的token>@github.com/<GitHub用户名>/<REPO>.git
```
