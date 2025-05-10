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
#include "include/base.h"
#include "include/server.h"
using namespace std;
const int Months = 12;

int main()
{
  Server sv = Server();
  sv.Run();
  // 使用arange构造一个一维向量，再用reshape变换到5x5的矩阵
  // vector<TaskState> v_ts;
  // TaskState ts = {"shanghai", false, -1};
  // v_ts.push_back(ts);
  // TaskState *pt_ts = &ts;
  // cout << pt_ts << endl;
  // cout << pt_ts->name << endl;
  // cout << pt_ts->is_running << endl;
  // cout <<sizeof(pt_ts) << endl;
  return 0;
}