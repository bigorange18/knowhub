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





