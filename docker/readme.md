# docker

docker三要素：

镜像、容器、仓库





## 虚悬镜像

这个镜像既没有仓库名，也没有标签，均为 `<none>`。：

```bash
#  命令来便捷的查看镜像、容器、数据卷所占用的空间。
docker system df
```



## 清理所有处于终止状态的容器

用 `docker container ls -a` 命令可以查看所有已经创建的包括终止状态的容器，如果数量太多要一个个删除可能会很麻烦，用下面的命令可以清理掉所有处于终止状态的容器。

```bash
docker container prune
```



## 启动容器

```bash
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
```

其中 *OPTIONS* 可以是以下几种：

*-d*：后台运行容器，并返回容器ID。

*-p*：指定端口映射，格式为：主机(宿主)端口:容器端口。

*-P*：随机端口映射，容器内部端口随机映射到主机的端口。

*-i*：以交互模式运行容器，通常与 *-t* 同时使用。

*-t*：为容器重新分配一个伪输入终端，通常与 *-i* 同时使用。

*--name*：为容器指定一个名称。

*-e*：设置环境变量。

*-v*：绑定一个卷。

```bash
docker run -p 80:80 -v /data:/data -d nginx:latest
```







## 容器指令



```bash
# 查看正在运行的容器
docker ps

# 查看所有容器，包括已经停止的容器
docker ps -a

# 停止容器
docker stop [container_id]

# 重启容器
docker restart [container_]
```

