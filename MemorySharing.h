#pragma once

#include <stdio.h>
#include <errno.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdlib.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

class MemorySharer
{
protected:
    static const char* GETKEYDIR;
    key_t key;
    int shmId;
    uchar* ptr;

    void ErrExit(const char* msg);

public:
    MemorySharer(const ushort projectId, uint shmSize);
    ~MemorySharer(void);
};

class ImageMemorySharer : public MemorySharer
{
public:
    uchar* getPtr(void);

    ImageMemorySharer(const ushort projectId, uint shmSize);
};

class ByteMemorySharer : public MemorySharer
{
public:
    ByteMemorySharer(const ushort projectId, uint shmSize = 1);
    uchar ReadByte(void);

    void WriteByte(uchar x);
};