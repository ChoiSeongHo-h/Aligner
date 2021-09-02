#include "MemorySharing.h"

const char* MemorySharer::GETKEYDIR = "/tmp";

MemorySharer::MemorySharer(const ushort projectId, uint shmSize)
{
    (this->key) = ftok(MemorySharer::GETKEYDIR, projectId);
    if ((this->key) < 0)
        ErrExit("ftok error");

    (this->shmId) = shmget((this->key), shmSize, IPC_CREAT | IPC_EXCL | 0664);
    if ((this->shmId) == -1)
    {
        if (errno == EEXIST)
        {
            printf("shared memeory already exist\n");
            (this->shmId) = shmget((this->key), 0, 0);
            printf("reference shmId = %d\n", (this->shmId));
        }
        else
        {
            perror("errno");
            ErrExit("shmget error");
        }
    }

    if ((void*)((this->ptr) = (uchar*)shmat((this->shmId), 0, 0)) == (void*)-1)
    {
        if (shmctl((this->shmId), IPC_RMID, NULL) == -1)
            ErrExit("shmctl error");
        else
        {
            printf("Attach shared memory failed\n");
            printf("remove shared memory identifier successful\n");
        }
        ErrExit("shmat error");
    }
    *ptr = 0;
}

MemorySharer::~MemorySharer(void)
{
    if (shmdt(this->ptr) < 0)
        ErrExit("shmdt error");

    if (shmctl((this->shmId), IPC_RMID, NULL) == -1)
        ErrExit("shmctl error");
    else
    {
        printf("Finally\n");
        printf("remove shared memory identifier successful\n");
    }
}

void MemorySharer::ErrExit(const char* msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

ImageMemorySharer::ImageMemorySharer(const ushort projectId, uint shmSize) : MemorySharer(projectId, shmSize)
{
}

ByteMemorySharer::ByteMemorySharer(const ushort projectId, uint shmSize) : MemorySharer(projectId, shmSize)
{
}

uchar* ImageMemorySharer::getPtr(void)
{
    return ptr;
}

uchar ByteMemorySharer::ReadByte(void)
{
    return *ptr;
}

void ByteMemorySharer::WriteByte(uchar x)
{
    *ptr = x;
}