/*
 * GILLIES Vulkan Benchmark - Scale test
 *
 * Testing GPU advantage at larger scales.
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// Validated SPIR-V for XOR: out[i] = a[i] + b[i] - 2.0 * a[i] * b[i]
static const uint32_t xor_shader_spirv[] = {
    0x07230203, 0x00010000, 0x00070000, 0x00000026,
    0x00000000, 0x00020011, 0x00000001, 0x0003000e,
    0x00000000, 0x00000001, 0x0006000f, 0x00000005,
    0x00000001, 0x6e69616d, 0x00000000, 0x00000002,
    0x00060010, 0x00000001, 0x00000011, 0x00000100,
    0x00000001, 0x00000001, 0x00040047, 0x00000002,
    0x0000000b, 0x0000001c, 0x00040047, 0x00000003,
    0x00000006, 0x00000004, 0x00050048, 0x00000004,
    0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x00000004, 0x00000003, 0x00040047, 0x00000005,
    0x00000022, 0x00000000, 0x00040047, 0x00000005,
    0x00000021, 0x00000000, 0x00040047, 0x00000006,
    0x00000006, 0x00000004, 0x00050048, 0x00000007,
    0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x00000007, 0x00000003, 0x00040047, 0x00000008,
    0x00000022, 0x00000000, 0x00040047, 0x00000008,
    0x00000021, 0x00000001, 0x00040047, 0x00000009,
    0x00000006, 0x00000004, 0x00050048, 0x0000000a,
    0x00000000, 0x00000023, 0x00000000, 0x00030047,
    0x0000000a, 0x00000003, 0x00040047, 0x0000000b,
    0x00000022, 0x00000000, 0x00040047, 0x0000000b,
    0x00000021, 0x00000002, 0x00020013, 0x0000000c,
    0x00030021, 0x0000000d, 0x0000000c, 0x00040015,
    0x0000000e, 0x00000020, 0x00000000, 0x00040017,
    0x0000000f, 0x0000000e, 0x00000003, 0x00040020,
    0x00000010, 0x00000001, 0x0000000f, 0x00040020,
    0x00000011, 0x00000001, 0x0000000e, 0x00030016,
    0x00000012, 0x00000020, 0x00040015, 0x00000013,
    0x00000020, 0x00000001, 0x0004002b, 0x00000013,
    0x00000014, 0x00000000, 0x0004002b, 0x00000012,
    0x00000015, 0x40000000, 0x0003001d, 0x00000003,
    0x00000012, 0x0003001d, 0x00000006, 0x00000012,
    0x0003001d, 0x00000009, 0x00000012, 0x0003001e,
    0x00000004, 0x00000003, 0x0003001e, 0x00000007,
    0x00000006, 0x0003001e, 0x0000000a, 0x00000009,
    0x00040020, 0x00000016, 0x00000002, 0x00000004,
    0x00040020, 0x00000017, 0x00000002, 0x00000007,
    0x00040020, 0x00000018, 0x00000002, 0x0000000a,
    0x00040020, 0x00000019, 0x00000002, 0x00000012,
    0x0004003b, 0x00000010, 0x00000002, 0x00000001,
    0x0004003b, 0x00000016, 0x00000005, 0x00000002,
    0x0004003b, 0x00000017, 0x00000008, 0x00000002,
    0x0004003b, 0x00000018, 0x0000000b, 0x00000002,
    0x00050036, 0x0000000c, 0x00000001, 0x00000000,
    0x0000000d, 0x000200f8, 0x0000001a, 0x00050041,
    0x00000011, 0x0000001b, 0x00000002, 0x00000014,
    0x0004003d, 0x0000000e, 0x0000001c, 0x0000001b,
    0x00060041, 0x00000019, 0x0000001d, 0x00000005,
    0x00000014, 0x0000001c, 0x0004003d, 0x00000012,
    0x0000001e, 0x0000001d, 0x00060041, 0x00000019,
    0x0000001f, 0x00000008, 0x00000014, 0x0000001c,
    0x0004003d, 0x00000012, 0x00000020, 0x0000001f,
    0x00050081, 0x00000012, 0x00000021, 0x0000001e,
    0x00000020, 0x00050085, 0x00000012, 0x00000022,
    0x0000001e, 0x00000020, 0x00050085, 0x00000012,
    0x00000023, 0x00000015, 0x00000022, 0x00050083,
    0x00000012, 0x00000024, 0x00000021, 0x00000023,
    0x00060041, 0x00000019, 0x00000025, 0x0000000b,
    0x00000014, 0x0000001c, 0x0003003e, 0x00000025,
    0x00000024, 0x000100fd, 0x00010038
};

#define VK_CHECK(call) do { \
    VkResult r = call; \
    if (r != VK_SUCCESS) { \
        fprintf(stderr, "VK error %d at line %d\n", r, __LINE__); \
        exit(1); \
    } \
} while(0)

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float cpu_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

uint32_t find_memory_type(VkPhysicalDevice physDev, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0;
}

int main() {
    printf("\n");
    printf("==============================================================================\n");
    printf("           GILLIES Vulkan Benchmark - Scale Test\n");
    printf("==============================================================================\n\n");

    // Setup Vulkan
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "GILLIES",
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo
    };
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, NULL);
    VkPhysicalDevice* devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &devCount, devs);
    VkPhysicalDevice physDev = devs[0];
    free(devs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDev, &props);
    printf("GPU: %s\n\n", props.deviceName);

    uint32_t qfCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
    VkQueueFamilyProperties* qfProps = malloc(qfCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);
    uint32_t queueFamily = 0;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    }
    free(qfProps);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queueFamily, .queueCount = 1, .pQueuePriorities = &prio
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo
    };
    VkDevice device;
    VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queueFamily,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, NULL, &cmdPool));

    // Create pipeline
    VkDescriptorSetLayoutBinding bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT }
    };
    VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings
    };
    VkDescriptorSetLayout descLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descLayoutInfo, NULL, &descLayout));

    VkPipelineLayoutCreateInfo pipeLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &descLayout
    };
    VkPipelineLayout pipeLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipeLayoutInfo, NULL, &pipeLayout));

    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(xor_shader_spirv), .pCode = xor_shader_spirv
    };
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(device, &shaderInfo, NULL, &shader));

    VkComputePipelineCreateInfo compInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader, .pName = "main"
        },
        .layout = pipeLayout
    };
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compInfo, NULL, &pipeline));

    VkDescriptorPoolSize poolSize = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 };
    VkDescriptorPoolCreateInfo descPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize
    };
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(device, &descPoolInfo, NULL, &descPool));

    // Test different scales
    int sizes[] = {65536, 262144, 1048576, 4194304, 16777216};
    int nSizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-12s %12s %12s %12s\n", "Elements", "GPU (M/s)", "CPU (M/s)", "GPU Speedup");
    printf("%-12s %12s %12s %12s\n", "--------", "--------", "--------", "-----------");

    for (int s = 0; s < nSizes; s++) {
        int N = sizes[s];
        VkDeviceSize bufSize = N * sizeof(float);

        // Create buffers
        VkBufferCreateInfo bufCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };

        VkBuffer bufA, bufB, bufOut;
        VK_CHECK(vkCreateBuffer(device, &bufCreateInfo, NULL, &bufA));
        VK_CHECK(vkCreateBuffer(device, &bufCreateInfo, NULL, &bufB));
        VK_CHECK(vkCreateBuffer(device, &bufCreateInfo, NULL, &bufOut));

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, bufA, &memReq);
        VkDeviceSize alignedSize = (memReq.size + memReq.alignment - 1) & ~(memReq.alignment - 1);

        VkMemoryAllocateInfo allocInfo = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = alignedSize * 3,
            .memoryTypeIndex = find_memory_type(physDev, memReq.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        };
        VkDeviceMemory memory;
        VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &memory));

        VK_CHECK(vkBindBufferMemory(device, bufA, memory, 0));
        VK_CHECK(vkBindBufferMemory(device, bufB, memory, alignedSize));
        VK_CHECK(vkBindBufferMemory(device, bufOut, memory, alignedSize * 2));

        float* mapped;
        VK_CHECK(vkMapMemory(device, memory, 0, alignedSize * 3, 0, (void**)&mapped));

        float* dataA = mapped;
        float* dataB = (float*)((char*)mapped + alignedSize);
        float* dataOut = (float*)((char*)mapped + alignedSize * 2);

        for (int i = 0; i < N; i++) {
            dataA[i] = (float)(i % 2);
            dataB[i] = (float)((i / 2) % 2);
            dataOut[i] = 0.0f;
        }

        // Setup descriptor set
        VkDescriptorSetAllocateInfo descAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descPool, .descriptorSetCount = 1, .pSetLayouts = &descLayout
        };
        VkDescriptorSet descSet;
        vkResetDescriptorPool(device, descPool, 0);
        VK_CHECK(vkAllocateDescriptorSets(device, &descAllocInfo, &descSet));

        VkDescriptorBufferInfo bufInfos[3] = {
            { .buffer = bufA, .offset = 0, .range = bufSize },
            { .buffer = bufB, .offset = 0, .range = bufSize },
            { .buffer = bufOut, .offset = 0, .range = bufSize }
        };
        VkWriteDescriptorSet writes[3];
        for (int i = 0; i < 3; i++) {
            writes[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descSet, .dstBinding = i, .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bufInfos[i]
            };
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, NULL);

        // Command buffer
        VkCommandBufferAllocateInfo cmdAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = cmdPool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VkCommandBuffer cmdBuf;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuf));

        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, NULL);
        vkCmdDispatch(cmdBuf, N / 256, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cmdBuf));

        VkFence fence;
        VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(device, &fenceInfo, NULL, &fence));

        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &cmdBuf
        };

        // GPU benchmark (multiple runs)
        double gpu_total = 0;
        int runs = 5;
        for (int r = 0; r < runs; r++) {
            vkResetFences(device, 1, &fence);
            double start = get_time();
            VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
            VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
            gpu_total += get_time() - start;
        }
        double gpu_time = gpu_total / runs;

        // CPU benchmark
        volatile float sink = 0.0f;
        float acc = 0.0f;
        double start = get_time();
        for (int i = 0; i < N; i++) {
            acc += cpu_xor(dataA[i], dataB[i]);
        }
        sink = acc;
        double cpu_time = get_time() - start;

        double gpu_mops = (N / gpu_time) / 1e6;
        double cpu_mops = (N / cpu_time) / 1e6;
        double speedup = gpu_mops / cpu_mops;

        printf("%-12d %12.2f %12.2f %12.2fx\n", N, gpu_mops, cpu_mops, speedup);

        // Cleanup
        vkDestroyFence(device, fence, NULL);
        vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuf);
        vkDestroyBuffer(device, bufA, NULL);
        vkDestroyBuffer(device, bufB, NULL);
        vkDestroyBuffer(device, bufOut, NULL);
        vkUnmapMemory(device, memory);
        vkFreeMemory(device, memory, NULL);
    }

    printf("\n==============================================================================\n");
    printf("  GILLIES Vulkan: Shapes running on Thor GPU!\n");
    printf("==============================================================================\n\n");

    // Cleanup
    vkDestroyDescriptorPool(device, descPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyPipelineLayout(device, pipeLayout, NULL);
    vkDestroyDescriptorSetLayout(device, descLayout, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    return 0;
}
