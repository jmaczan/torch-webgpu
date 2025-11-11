#pragma once
#include <iostream>

namespace torch_webgpu
{
    inline void replace_string(std::string &src, const std::string &from, const std::string &to)
    {
        size_t pos = 0;
        while ((pos = src.find(from, pos)) != std::string::npos)
        {
            src.replace(pos, from.size(), to);
            pos += to.size();
        }
    }
}
