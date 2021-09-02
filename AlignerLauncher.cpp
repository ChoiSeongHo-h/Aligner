#include "Aligner.h"

int main(int argc, char** argv)
{
    Aligner aligner;

    int status;
    if (fork() == 0)
    {
        system("python3 ../AlignerViewer/manage.py runserver");
        exit(0);
    }

    aligner.Launch();

    wait(&status);
    return 0;
}