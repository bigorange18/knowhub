#include <string>

#ifndef SERVER_H_
#define SERVER_H_
#endif
#define Port 8888


struct state{
    std::string name;
};



class Server{


public:
    Server(){};
    ~Server(){};
    void Run();

    public:
        std::string server_name_;

    private:
        bool is_working;


};