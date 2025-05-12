

## Git

Git 是一个分布式版本控制系统，用于跟踪代码的变更。它最初由 Linus Torvalds 开发，用于管理 Linux 内核的开发。Git 在软件开发中非常流行，因为它提供了强大的工具来处理代码的版本控制、分支管理、合并冲突等。



### 初始化代码仓

```bash
#初始化文件夹
git init
#给git添加邮箱、密码
git config --global user.email "xxx"
git config --global user.name "xxx"
#查看当前文件中文件状态，是否添加、提交
git commit -m "说明"

git clone 仓库地址
#git克隆分支
git clone -b 分支名 仓库地址
```



### GIt名词解释

| 名称                            | 描述                                                   |
| ------------------------------- | ------------------------------------------------------ |
| 工作区(workspace)               | 电脑中代码库目录，就是文件                             |
| 暂缓区(stage)                   | 用于临时存放修改的文件，实际上是一个(.git/index)的文件 |
| 版本库/仓库(Repository)         | Git的管理仓库，管理版本的数据库                        |
| 服务端/远程仓库(origin/ remote) | 服务端的版本库                                         |

### Git配置文件

 - **系统全局配置(--system)**: 包含适用于系统所有用户和所有仓库的配置信息；
 - **用户全局配置（--global）**:当前系统用户的全局配置；
 - **仓库/项目配置（--local）**：

```bash
#查看git配置
git config --list
git config -l

#查看系统配置

#查看当前用户配置
git config --list --global

#查看当前仓库配置信息
git config --local --list
```



### 添加代理下载git代码仓

```bash
git clone https://ghp.ci/你需要下载的代码仓
使用代理下载eg:
git clone https://ghp.ci/https://github.com/bigorange18/knowhub.git
```




### 使用token提交代码

1.已经有仓库:为仓库添加token

```shell

git remote set-url origin https://<你的token>@github.com/<GitHub用户名>/<要提交到的库名>.git
```


```shell
2.原本没有仓库: git clone 添加token

git clone https://<你的token>@github.com/<GitHub用户名>/<REPO>.git
```





### .gitignore

```
#提交时候，需要忽略的文件，在 .gitignore文件中
#1、忽略文件中的空行或以井号（#）开始的行将会被忽略。
#2、可以使用Linux通配符。例如：星号（*）代表任意多个字符，问号（？）代表一个字符，方括号##（[abc]）代表可选字符范围，大括号（{string1,string2,...}）代表可选的字符串等。
#3、如果名称的最前面有一个感叹号（!），表示例外规则，将不被忽略。
#4、如果名称的最前面是一个路径分隔符（/），表示要忽略的文件在此目录下，而子目录中的文件不忽略。
#5、如果名称的最后面是一个路径分隔符（/），表示要忽略的是此目录下该名称的子目录，而非文件（默认文件或目录都忽略）。
#例子
*.txt #忽略所有的txt文件
!lib.txt #lib.txt文件除外
/temp # 仅忽略项目根目录下的TODO文件,不包括其它目录temp
build/       #忽略build/目录下的所有文件
doc/*.txt    #会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```

## Git常用指令

### git branch

  1. 查看本地所有分支：

```bash
git branch
```

  2. 查看远程分支

```bash
git branch -r
```

  3. 查看远程本地和远程所有分支

```bash
git branch -a
```

  4. 新建分支

```bash
git branch 分支名称
```

  5. 删除本地分支

```bash
git branch -d 分支名称
```

  6. 删除远程分支，并推送到远程服务

```bash
git branch -d 分支名
git push origin:分支名
```

  7. 重命名分支

```bash
git branch -m 新分支名 老分支名
```

### Git 提交

```bash
git add .
git commit -m "s"
git push
```

### git log

1、查看所有提交的日志记录

```bash
git log 
```

### Git fetch用法

**git fetch**是将远程主机的最新内容拉到本地，用户检查后确定哪些内容合入。**git pull**是将远程主机的最新内容拉下后直接合并，**git pull=git fetch+git merge**。

  1. 拉取远程仓库最新内容

```bash
git fetch origin
```

  2. 拉取远程仓库master分支内容

```bash
git fetch origin master
```

### Git撤回

 #### 1.git checkout使用

```bash
#1.撤销工作区（未暂缓）的修改，把暂缓区恢复到工作区。
git checkout .

#2.撤销某个文件的修改
git checkout filename

#3. 撤销工作区、暂缓区的修改，用**HEAD**指向当前分支最新版本替换工作区、暂缓区
git checkout HEAD .

#4. 撤销工作区、暂缓区的修改，用**HEAD**指向当前分支最新版本替换工作区、暂缓区
git checkout HEAD filename
```

 #### 2. git reset

```bash
#1. 撤销暂缓区状态，同git reset HEAD, 不影响工作区
git reset 

#2. 指定文件
git reset HEAD filename

#3.回退到指定版本，清空暂缓区，不影响工作区
git reset commit

#4.

```

| 模式          | 描述                             | HEAD位置 | 暂缓区 | 工作区 |
| ------------- | -------------------------------- | -------- | ------ | ------ |
| soft          | 回退到某一个版本，工作区不变     | 修改     | 不修改 | 不修改 |
| mixed（默认） | 撤销暂缓区工作状态，不影响工作区 | 修改     | 修改   | 不修改 |
| hard          | 重置未提交修改                   | 修改     | 修改   | 修改   |




 #### 3. git revert

 安全的撤销某一个提交记录，基本原理就是生产一个新的提交，用原提交的逆向操作来完成撤销操作。注意，这不同于reset，reset是回退版本，revert只是用于撤销某一次历史提交，操作是比较安全的。

#### 4.checkout/reset/revert总结

| 指令     | 主要作用                       | 撤回工作区       | 撤回工作区、暂缓区     | 回退版本                | 安全性                     |
| -------- | ------------------------------ | ---------------- | ---------------------- | ----------------------- | -------------------------- |
| checkout | 撤回工作区、暂缓区未提交的修改 | git checkout .   | git checkout HEAD .    | /                       | 只针对未提交修改           |
| reset    | 回退版本，重置工作区、暂缓区   | git reset HEAD . | git rest --hard HEAD . | git reset --hard commit | 回退已经push的内容，不安全 |
| revert   | 撤销某一次提交                 | /                | /                      | /                       | 安全                       |

### git stash

当你正在dev分支开发一个功能时，代码写了一半，突然有一个线上的bug急需要马上修改。dev分支Bug没写完，不方便提交，就不能切换到主分支去修复线上bug。Git提供一个stash功能，可以把当前工作区、暂存区 未提交的内容“隐藏”起来，就像什么都没发生一样。

| 指令                          | 功能                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| git stash                     | 把未提交的内容隐藏起来，包括未暂存、已暂存，方便后续恢复工作环境 |
| git stash list                | 查看所有被隐藏的内容列表                                     |
| git stash pop                 | 恢复被隐藏的内容，通俗删除隐藏记录                           |
| git stash save "增加保存说明" | 可以增加保存说明                                             |
| git stash apply               | 恢复被隐藏的文件，但是隐藏记录不删除                         |
| git stash drop                | 删除隐藏记录                                                 |


```bash
# 查看当前隐藏内容
git stash list
#随便改一下代码
# 隐藏
git stash
# 保存工作目录和索引状态 WIP on only: 760fc4f s
 
# 查看被隐藏的内容
git stash list
# stash@{0}: WIP on only: 760fc4f s
 
# 比较一下，什么都没有，一切都没有发生过！
git diff
 
# 去其他分支修改bug，修复完成回到当前分支，恢复工作区
git stash pop
#再次查看隐藏内容
git stash list
#输出为空

```

## git lfs
