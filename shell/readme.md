# Shell脚本



## 简介

Shell 编程跟 JavaScript、php 编程一样，只要有一个能编写代码的文本编辑器和一个能解释执行的脚本解释器就可以了。

## 常用方法

### 变量

变量名的命名须遵循如下规则：

- 变量名和等号之间不能有空格；
- **只包含字母、数字和下划线：** 变量名可以包含字母（大小写敏感）、数字和下划线 **_**，不能包含其他特殊字符；
- **不能以数字开头：** 变量名不能以数字开头，但可以包含数字。
- **避免使用 Shell 关键字：** 不要使用Shell的关键字（例如 if、then、else、fi、for、while 等）作为变量名，以免引起混淆。
- **使用大写字母表示常量：** 习惯上，常量的变量名通常使用大写字母，例如 **PI=3.14**。
- **避免使用特殊符号：** 尽量避免在变量名中使用特殊符号，因为它们可能与 Shell 的语法产生冲突。
- **避免使用空格：** 变量名中不应该包含空格，因为空格通常用于分隔命令和参数。

eg:

```bash
name="chen"
age=18
```

### 变量类型

#### 字符串变量

```sh
str='this is a string'
```



#### 整数变量

```sh
# 整数变量
declare -i my_integer=42
```



#### 数组变量

```sh

# 数组变量
my_array=(1 2 3 4 5)
# 读取数据内容
${my_array[idx]}
#读取数据所有内容
${my_array[*]}
${my_array[@]}


declare -A associative_array
associative_array["name"]="John"
associative_array["age"]=30


```



#### 环境变量
```shell
echo $PATH

# 特殊变量
$0  # 当前文件名
$1	# 传递第一个参数
$2	# 传递第二个参数
$#  # 获取参数个数
$?  # 获取上次执行的结果，执行成功为0，其他数字为执行失败
$*	# 所有参数
$@  # 所有参数
$$  # 当前文件的pid
$_	# 获取上次传入的最后一个参数
```





### 变量使用

使用一个定义过的变量，只要在变量名前面加美元符号即可，量名外面的花括号是可选的，加不加都行，如：

```bash
name="chen"
echo $name
echo ${name}
```

**只读变量**

使用 readonly 命令可以将变量定义为只读变量，只读变量的值不能被改变。

```bash
name="chen"
readonly name
```







**删除变量**

使用 unset 命令可以删除变量

```bash
unset variable_name
```





### 运算符

| 运算符 |                     说明                      |             举例              |
| :----: | :-------------------------------------------: | :---------------------------: |
|   +    |                     加法                      |  `expr $a + $b` 结果为 30。   |
|   -    |                     减法                      |  `expr $a - $b` 结果为 -10。  |
|   *    |                     乘法                      | `expr $a \* $b` 结果为  200。 |
|   /    |                     除法                      |   `expr $b / $a` 结果为 2。   |
|   %    |                     取余                      |   `expr $b % $a` 结果为 0。   |
|   =    |                     赋值                      |  a=$b 把变量 b 的值赋给 a。   |
|   ==   |   相等。用于比较两个数字，相同则返回 true。   |   [ $a == $b ] 返回 false。   |
|   !=   | 不相等。用于比较两个数字，不相同则返回 true。 |   [ $a != $b ] 返回 true。    |
|        |                                               |                               |







| 参数 | 说明           |      |
| ---- | -------------- | ---- |
| -eq  | 等于则为真     |      |
| -ne  | 不等于则为真   |      |
| -gt  | 大于则为真     |      |
| -ge  | 大于等于则为真 |      |
| -lt  | 小于则为真     |      |
| -le  | 小于等于则为真 |      |
|      |                |      |
|      |                |      |
|      |                |      |





### 控制流

```sh
if condation
then
	command
if
```



```sh
if condation
then
	command1
elif
	command2
	
else
	command3
fi
```







## 函数



```sh
[function] funname[()]
{

	action;
	return ${xx};

}
```



### 函数传参



调用函数时可以向其传递参数。在函数体内部，通过 $n 的形式来获取参数的值，例如，$1表示第一个参数，$2表示第二个参数...

| 参数处理 | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| $#       | 传递到脚本或函数的参数个数                                   |
| $*       | 以一个单字符串显示所有向脚本传递的参数                       |
| $$       | 脚本运行的当前进程ID号                                       |
| $!       | 后台运行的最后一个进程的ID号                                 |
| $@       | 与$*相同，但是使用时加引号，并在引号中返回每个参数。         |
| $-       | 显示Shell使用的当前选项，与set命令功能相同。                 |
| $?       | 显示最后命令的退出状态。0表示没有错误，其他任何值表明有错误。 |



```sh
my_function(){
	echo "第一个参数" $1
}
```





## shell 输出/输入重定向

大多数 UNIX 系统命令从你的终端接受输入并将所产生的输出发送回到您的终端。一个命令通常从一个叫标准输入的地方读取输入，默认情况下，这恰好是你的终端。同样，一个命令通常将其输出写入到标准输出，默认情况下，这也是你的终端。

### 重定向命令

| 命令            | 说明                                               |
| --------------- | -------------------------------------------------- |
| command > file  | 将输出重定向到 file。                              |
| command < file  | 将输入重定向到 file。                              |
| command >> file | 将输出以追加的方式重定向到 file。                  |
| n > file        | 将文件描述符为 n 的文件重定向到 file。             |
| n >> file       | 将文件描述符为 n 的文件以追加的方式重定向到 file。 |
| n >& m          | 将输出文件 m 和 n 合并。                           |
| n <& m          | 将输入文件 m 和 n 合并。                           |
| << tag          | 将开始标记 tag 和结束标记 tag 之间的内容作为输入。 |



### 输入重定向





### 输出重定向







## linux文件路径

/是目录层的分隔、表示符。只有一个/表明是root，/etc/表明是根目录下面的etc目录（当然目录最后不需要/，但有/直接表明他是目录，没有末尾的/，那么/etc需要检测一下确定是目录还是文件，虽然习惯上/etc绝对是目录）
~是一个代位符，表明的是个人目录的地址，因为每个用户都有自己的个人目录地址，所以用~作为统一替代这个根据用户不同而不同但有规可循的地址，来保证某些情况下的兼容问题。
如果以root账号登陆
~代表/root/
如果以name登陆
~代表/home/name/

EOF: end of file

EOL: end of line

EOB: end of block





## bash

```bash
# 清空终端中的历史记录
history -c

# 恢复历史记录
history -r
# 查看保存历史数据多少
echo $HISTSIZE
# 快速清屏
ctrl + l
```

## shell变量



### echo



| 参数 |                             作用                             |
| :--: | :----------------------------------------------------------: |
|  -n  | `echo`输出的文本末尾会有一个回车符。`-n`参数可以取消末尾的回车符，使得下一个提示符紧跟在输出内容的后面。 |
|  -e  | `-e`参数会解释引号（双引号和单引号）里面的特殊字符（比如换行符`\n`）。如果不使用`-e`参数，即默认情况下，引号会让特殊字符变成普通字符，`echo`不解释它们，原样输出。 |
|      |                                                              |







1. 单引号变量不识别特殊符号
2. 双引号变量能识别特殊符号

```bash
# 特殊变量
$0  # 当前文件名
$1	# 传递第一个参数
$2	# 传递第二个参数
$#  # 获取参数个数
$?  # 获取上次执行的结果，执行成功为0，其他数字为执行失败
$*	# 所有参数
$@  # 所有参数
$$  # 当前文件的pid
$_	# 获取上次传入的最后一个参数
```

```
${变量}			#返回变量值
${#变量}			#返回变量长度
${变量:start}		#返回变量start之后的字符
${变量:start:length}#返回start之后length长度的字符
${变量#word}		#从变量开头变量，删除最短匹配的word子串
${变量##word}
${%word}
${%%word}
${/pattern/string}
${//pattern/string}
```



删除变量

使用unset删除变量

```bash
unset varibale_name
```

## 快捷键

- `Ctrl + L`：清除屏幕并将当前行移到页面顶部。
- `Ctrl + C`：中止当前正在执行的命令。
- `Ctrl + U`：从光标位置删除到行首。
- `Ctrl + K`：从光标位置删除到行尾。
- `Ctrl + W`：删除光标位置前一个单词。
- `Ctrl + D`：关闭 Shell 会话。
- `↑`，`↓`：浏览已执行命令的历史记录。

## 