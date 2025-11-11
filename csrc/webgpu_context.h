#pragma once
#include <webgpu/webgpu_cpp.h>
#include <iostream>

namespace torch_webgpu
{
    struct WebGPUContext
    {
        wgpu::Instance instance;
        wgpu::Adapter adapter;
        wgpu::Device device;
        wgpu::Queue queue;

        WebGPUContext();

        wgpu::Instance getInstance();
        wgpu::Device getDevice();
        wgpu::Queue getQueue();
    };

    WebGPUContext &getWebGPUContext();
}