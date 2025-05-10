/*
 * @Author: bigorange chenorange2219@gmail.com
 * @Date: 2025-05-02 20:58:45
 * @LastEditors: bigorange chenorange2219@gmail.com
 * @LastEditTime: 2025-05-05 10:17:47
 * @FilePath: /webserver/main.cc
 * @Description:
 *
 * Copyright (c) 2025 by ${git_name_email}, All Rights Reserved.
 */

#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <gflags/gflags.h>
#include "lib/base.h"
using namespace std;
const int Months = 12;

int main()
{
  // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
  const float hight = 1.89e2;
  std::cout << "身高:" << hight << std::endl;
  std::cout << "\aOperation \"Hyperhype\" is now activated:" << std::endl;
  float week[7] = {1., 2, 3, 4, 5, 6, 7};
  week[4] = 100;
  std::cout << week[1] << std::endl;
  std::cout << sizeof(week) << std::endl;
  char name[20];
  string my_name;
  cin.getline(name, 20);
  std::cout << name << std::endl;
  cout << "Please enter your name:" << endl;
  cin >> my_name;
  cout << "hell," << my_name << endl;
  return 0;
}