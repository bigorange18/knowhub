#include <iostream>
#include <fstream>
#include <time.h>
#include "include/server.h"


namespace GameViPer{
template <class AnyType>
void swap(AnyType &a, AnyType &b){
    AnyType t;
    t = a;
    a = b;
    b = t;
}

void test(){
    std::ofstream outfile;
    outfile.open("../data/1.txt");
    if(!outfile.is_open()){
        std::cout << "open file fail !" << std::endl;
    }
    std::cout << "Please write some words into txt:" << std::endl;
    char save_[20];
    std::cin.getline(save_, 20);
    outfile << save_;
    outfile.close();

}

void test2(){
    int *pt = new int;
    *pt = 100;
    std::cout << *pt <<std::endl;
};

void test3(){
    std::ifstream inFile;
    inFile.open("../data/1.txt");
    if(!inFile.is_open()){
        std::cout << "open file fail !" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string word;
    inFile >> word;
    std::cout <<word << std::endl;
    int count = 0;
    while (inFile.good())
    {
        ++count;
        inFile >> word;
        std::cout <<word << std::endl;
    }
    if (inFile.eof()){
        std::cout << "文件已经读完了" <<std::endl;
    }
    inFile.close();
    
}

void test4(){
    // time start_t;
    double a = 4;
    double b = 6;
    swap(a, b);
    std::cout << a << b << std::endl;
    
}

Server::Server(){};

Server::Server(std::string name){
    server_name_ = name;
}


void Server::Run(){
    is_working = true;
    // test();
    // test3();
    test4();

};

} // namespace GameViper