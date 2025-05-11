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

### 1、添加代理下载git代码仓

```bash
git clone https://ghp.ci/你需要下载的代码仓
eg:
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

