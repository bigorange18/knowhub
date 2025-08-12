###
 # @Author: chenorange chenorange2219@gmail.com
 # @Date: 2025-08-12 07:45:41
 # @LastEditors: chenorange chenorange2219@gmail.com
 # @LastEditTime: 2025-08-12 07:58:05
 # @FilePath: \knowhub\shell\knowhub.sh
 # @Description: 
 # 
 # Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
### 
echo $(pwd)

declare -i myage=18
declare -A fruits
fruits['apple']=4.56
fruits['banana']=2.5

echo ${fruits[@]}
echo ${!fruits[@]}
for  key in ${!fruits[@]} 
do
    echo $key:${fruits[$key]}
done


for key in ${pwd}/*
do 
    if [ -f ${key} ]; then
        echo ${key}
    fi
done