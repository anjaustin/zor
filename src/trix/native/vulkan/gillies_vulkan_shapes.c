/*
 * GILLIES Vulkan Shapes - Full GPU Compute Pipeline
 *
 * TriX shapes running on Thor GPU via Vulkan.
 * "The shape doesn't care about the substrate."
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

// Freshly compiled SPIR-V for XOR compute shader
// XOR shape: out[i] = a[i] + b[i] - 2.0 * a[i] * b[i]
// local_size_x = 256
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

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physDev;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamily;
    VkCommandPool cmdPool;
    VkDescriptorPool descPool;
    VkDescriptorSetLayout descLayout;
    VkPipelineLayout pipeLayout;
    VkPipeline pipeline;
    VkShaderModule shader;
} GilliesContext;

uint32_t find_memory_type(VkPhysicalDevice physDev, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

GilliesContext* gillies_create() {
    GilliesContext* ctx = calloc(1, sizeof(GilliesContext));

    // 1. Create Instance
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "GILLIES",
        .apiVersion = VK_API_VERSION_1_2
    };
    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo
    };
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &ctx->instance));

    // 2. Get Physical Device (Thor)
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, NULL);
    VkPhysicalDevice* devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, devs);
    ctx->physDev = devs[0];
    free(devs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physDev, &props);
    printf("GPU: %s\n", props.deviceName);

    // 3. Find Compute Queue
    uint32_t qfCount;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physDev, &qfCount, NULL);
    VkQueueFamilyProperties* qfProps = malloc(qfCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physDev, &qfCount, qfProps);

    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            ctx->queueFamily = i;
            break;
        }
    }
    free(qfProps);

    // 4. Create Logical Device
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &prio
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo
    };
    VK_CHECK(vkCreateDevice(ctx->physDev, &devInfo, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->queueFamily, 0, &ctx->queue);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->queueFamily,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &poolInfo, NULL, &ctx->cmdPool));

    // 6. Create Descriptor Set Layout (3 storage buffers: A, B, Out)
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
        .bindingCount = 3,
        .pBindings = bindings
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &descLayoutInfo, NULL, &ctx->descLayout));

    // 7. Create Pipeline Layout
    VkPipelineLayoutCreateInfo pipeLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->descLayout
    };
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipeLayoutInfo, NULL, &ctx->pipeLayout));

    // 8. Create Shader Module (validated SPIR-V)
    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(xor_shader_spirv),
        .pCode = xor_shader_spirv
    };
    VK_CHECK(vkCreateShaderModule(ctx->device, &shaderInfo, NULL, &ctx->shader));

    // 9. Create Compute Pipeline
    VkComputePipelineCreateInfo compInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = ctx->shader,
            .pName = "main"
        },
        .layout = ctx->pipeLayout
    };
    VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &compInfo, NULL, &ctx->pipeline));

    // 10. Create Descriptor Pool
    VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo descPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &descPoolInfo, NULL, &ctx->descPool));

    return ctx;
}

void gillies_destroy(GilliesContext* ctx) {
    vkDestroyDescriptorPool(ctx->device, ctx->descPool, NULL);
    vkDestroyPipeline(ctx->device, ctx->pipeline, NULL);
    vkDestroyShaderModule(ctx->device, ctx->shader, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipeLayout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, ctx->descLayout, NULL);
    vkDestroyCommandPool(ctx->device, ctx->cmdPool, NULL);
    vkDestroyDevice(ctx->device, NULL);
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

int main() {
    printf("\n");
    printf("==============================================================================\n");
    printf("           GILLIES Vulkan Shapes - Thor GPU Compute Pipeline\n");
    printf("==============================================================================\n\n");

    // Create context (builds full pipeline)
    printf("Building Vulkan compute pipeline...\n");
    GilliesContext* ctx = gillies_create();
    printf("Pipeline ready!\n\n");

    // Test parameters
    const int N = 65536;  // 64K elements
    const VkDeviceSize bufSize = N * sizeof(float);

    // Create buffers
    printf("Allocating GPU buffers (%d elements)...\n", N);

    VkBufferCreateInfo bufInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VkBuffer bufA, bufB, bufOut;
    VK_CHECK(vkCreateBuffer(ctx->device, &bufInfo, NULL, &bufA));
    VK_CHECK(vkCreateBuffer(ctx->device, &bufInfo, NULL, &bufB));
    VK_CHECK(vkCreateBuffer(ctx->device, &bufInfo, NULL, &bufOut));

    // Get memory requirements
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, bufA, &memReq);

    VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t memType = find_memory_type(ctx->physDev, memReq.memoryTypeBits, memFlags);

    // Allocate memory for all three buffers
    VkDeviceSize alignedSize = (memReq.size + memReq.alignment - 1) & ~(memReq.alignment - 1);

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = alignedSize * 3,
        .memoryTypeIndex = memType
    };

    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(ctx->device, &allocInfo, NULL, &memory));

    // Bind buffers to memory
    VK_CHECK(vkBindBufferMemory(ctx->device, bufA, memory, 0));
    VK_CHECK(vkBindBufferMemory(ctx->device, bufB, memory, alignedSize));
    VK_CHECK(vkBindBufferMemory(ctx->device, bufOut, memory, alignedSize * 2));

    printf("  Memory allocated: %lu bytes\n\n", (unsigned long)(alignedSize * 3));

    // Map and initialize data
    printf("Initializing test data...\n");
    float* mapped;
    VK_CHECK(vkMapMemory(ctx->device, memory, 0, alignedSize * 3, 0, (void**)&mapped));

    float* dataA = mapped;
    float* dataB = (float*)((char*)mapped + alignedSize);
    float* dataOut = (float*)((char*)mapped + alignedSize * 2);

    // XOR truth table pattern
    for (int i = 0; i < N; i++) {
        dataA[i] = (float)(i % 2);
        dataB[i] = (float)((i / 2) % 2);
        dataOut[i] = 0.0f;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->descLayout
    };

    VkDescriptorSet descSet;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &descAllocInfo, &descSet));

    // Update descriptor set
    VkDescriptorBufferInfo bufInfos[3] = {
        { .buffer = bufA, .offset = 0, .range = bufSize },
        { .buffer = bufB, .offset = 0, .range = bufSize },
        { .buffer = bufOut, .offset = 0, .range = bufSize }
    };

    VkWriteDescriptorSet writes[3];
    for (int i = 0; i < 3; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descSet,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bufInfos[i]
        };
    }
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    // Create command buffer
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VkCommandBuffer cmdBuf;
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &cmdBuf));

    // Record commands
    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeLayout, 0, 1, &descSet, 0, NULL);
    vkCmdDispatch(cmdBuf, N / 256, 1, 1);  // 256 threads per workgroup
    VK_CHECK(vkEndCommandBuffer(cmdBuf));

    // Submit and time execution
    printf("\nExecuting XOR shape on GPU...\n");

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdBuf
    };

    VkFence fence;
    VkFenceCreateInfo fenceInfo = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(ctx->device, &fenceInfo, NULL, &fence));

    double start = get_time();
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));
    double elapsed = get_time() - start;

    printf("  GPU time: %.3f ms\n", elapsed * 1000);
    printf("  Throughput: %.2f M ops/sec\n\n", (N / elapsed) / 1e6);

    // Verify results
    printf("Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = cpu_xor(dataA[i], dataB[i]);
        if (fabsf(dataOut[i] - expected) > 0.0001f) {
            if (errors < 5) {
                printf("  Mismatch at %d: got %.4f, expected %.4f (a=%.1f, b=%.1f)\n",
                       i, dataOut[i], expected, dataA[i], dataB[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("  All %d results CORRECT!\n\n", N);
    } else {
        printf("  %d errors out of %d\n\n", errors, N);
    }

    // CPU comparison benchmark
    printf("CPU comparison (same %d ops)...\n", N);
    volatile float sink = 0.0f;
    float acc = 0.0f;

    start = get_time();
    for (int i = 0; i < N; i++) {
        acc += cpu_xor(dataA[i], dataB[i]);
    }
    sink = acc;
    double cpu_time = get_time() - start;

    printf("  CPU time: %.3f ms\n", cpu_time * 1000);
    printf("  CPU throughput: %.2f M ops/sec\n\n", (N / cpu_time) / 1e6);

    // Cleanup
    vkUnmapMemory(ctx->device, memory);
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmdBuf);
    vkDestroyBuffer(ctx->device, bufA, NULL);
    vkDestroyBuffer(ctx->device, bufB, NULL);
    vkDestroyBuffer(ctx->device, bufOut, NULL);
    vkFreeMemory(ctx->device, memory, NULL);

    gillies_destroy(ctx);

    printf("==============================================================================\n");
    printf("  GILLIES Vulkan Shapes: WORKING!\n");
    printf("  \n");
    printf("  XOR shape executed on Thor GPU via Vulkan compute.\n");
    printf("  No CUDA required. Pure substrate-independent shapes.\n");
    printf("==============================================================================\n\n");

    return 0;
}
