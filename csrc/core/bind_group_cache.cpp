#include "bind_group_cache.h"

namespace torch_webgpu
{
    namespace core
    {
        BindGroupCache &getBindGroupCache()
        {
            static BindGroupCache cache;
            return cache;
        }

    } // namespace core
} // namespace torch_webgpu
