/*
 * GILLIES Vulkan Library - Python-callable shared library
 *
 * Exposes GILLIES Vulkan compute to Python via ctypes.
 * "17 billion ops/sec. From Python."
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// XOR shader SPIR-V (validated, working)
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
    if (r != VK_SUCCESS) return -1; \
} while(0)

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

    // Persistent buffers for batch processing
    VkBuffer bufA, bufB, bufOut;
    VkDeviceMemory memory;
    VkDescriptorSet descSet;
    VkCommandBuffer cmdBuf;
    float* mapped;
    size_t bufferSize;
    size_t alignedSize;
} GilliesContext;

static uint32_t find_memory_type(VkPhysicalDevice physDev, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0;
}

// ============================================================================
// PUBLIC API
// ============================================================================

GilliesContext* gillies_create(size_t max_elements) {
    GilliesContext* ctx = calloc(1, sizeof(GilliesContext));
    if (!ctx) return NULL;

    ctx->bufferSize = max_elements * sizeof(float);

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
    if (vkCreateInstance(&instInfo, NULL, &ctx->instance) != VK_SUCCESS) {
        free(ctx);
        return NULL;
    }

    // 2. Get Physical Device
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, NULL);
    if (devCount == 0) { free(ctx); return NULL; }
    VkPhysicalDevice* devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, devs);
    ctx->physDev = devs[0];
    free(devs);

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
    if (vkCreateDevice(ctx->physDev, &devInfo, NULL, &ctx->device) != VK_SUCCESS) {
        free(ctx);
        return NULL;
    }
    vkGetDeviceQueue(ctx->device, ctx->queueFamily, 0, &ctx->queue);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->queueFamily,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    vkCreateCommandPool(ctx->device, &poolInfo, NULL, &ctx->cmdPool);

    // 6. Create Descriptor Set Layout
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
    vkCreateDescriptorSetLayout(ctx->device, &descLayoutInfo, NULL, &ctx->descLayout);

    // 7. Create Pipeline Layout
    VkPipelineLayoutCreateInfo pipeLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->descLayout
    };
    vkCreatePipelineLayout(ctx->device, &pipeLayoutInfo, NULL, &ctx->pipeLayout);

    // 8. Create Shader Module
    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(xor_shader_spirv),
        .pCode = xor_shader_spirv
    };
    vkCreateShaderModule(ctx->device, &shaderInfo, NULL, &ctx->shader);

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
    vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &compInfo, NULL, &ctx->pipeline);

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
    vkCreateDescriptorPool(ctx->device, &descPoolInfo, NULL, &ctx->descPool);

    // 11. Create persistent buffers
    VkBufferCreateInfo bufInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = ctx->bufferSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    vkCreateBuffer(ctx->device, &bufInfo, NULL, &ctx->bufA);
    vkCreateBuffer(ctx->device, &bufInfo, NULL, &ctx->bufB);
    vkCreateBuffer(ctx->device, &bufInfo, NULL, &ctx->bufOut);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, ctx->bufA, &memReq);
    ctx->alignedSize = (memReq.size + memReq.alignment - 1) & ~(memReq.alignment - 1);

    VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t memType = find_memory_type(ctx->physDev, memReq.memoryTypeBits, memFlags);

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = ctx->alignedSize * 3,
        .memoryTypeIndex = memType
    };
    vkAllocateMemory(ctx->device, &allocInfo, NULL, &ctx->memory);

    vkBindBufferMemory(ctx->device, ctx->bufA, ctx->memory, 0);
    vkBindBufferMemory(ctx->device, ctx->bufB, ctx->memory, ctx->alignedSize);
    vkBindBufferMemory(ctx->device, ctx->bufOut, ctx->memory, ctx->alignedSize * 2);

    vkMapMemory(ctx->device, ctx->memory, 0, ctx->alignedSize * 3, 0, (void**)&ctx->mapped);

    // 12. Allocate descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->descLayout
    };
    vkAllocateDescriptorSets(ctx->device, &descAllocInfo, &ctx->descSet);

    // Update descriptor set
    VkDescriptorBufferInfo bufInfos[3] = {
        { .buffer = ctx->bufA, .offset = 0, .range = ctx->bufferSize },
        { .buffer = ctx->bufB, .offset = 0, .range = ctx->bufferSize },
        { .buffer = ctx->bufOut, .offset = 0, .range = ctx->bufferSize }
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descSet,
          .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufInfos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descSet,
          .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufInfos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descSet,
          .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufInfos[2] }
    };
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    // 13. Allocate command buffer
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmdPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &ctx->cmdBuf);

    return ctx;
}

void gillies_destroy(GilliesContext* ctx) {
    if (!ctx) return;

    vkUnmapMemory(ctx->device, ctx->memory);
    vkFreeMemory(ctx->device, ctx->memory, NULL);
    vkDestroyBuffer(ctx->device, ctx->bufA, NULL);
    vkDestroyBuffer(ctx->device, ctx->bufB, NULL);
    vkDestroyBuffer(ctx->device, ctx->bufOut, NULL);
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

int gillies_xor(GilliesContext* ctx, const float* a, const float* b, float* out, size_t n) {
    if (!ctx || n * sizeof(float) > ctx->bufferSize) return -1;

    // Copy input data
    float* dataA = ctx->mapped;
    float* dataB = (float*)((char*)ctx->mapped + ctx->alignedSize);
    memcpy(dataA, a, n * sizeof(float));
    memcpy(dataB, b, n * sizeof(float));

    // Record command buffer
    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkResetCommandBuffer(ctx->cmdBuf, 0);
    vkBeginCommandBuffer(ctx->cmdBuf, &beginInfo);

    vkCmdBindPipeline(ctx->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(ctx->cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeLayout, 0, 1, &ctx->descSet, 0, NULL);
    vkCmdDispatch(ctx->cmdBuf, (n + 255) / 256, 1, 1);

    vkEndCommandBuffer(ctx->cmdBuf);

    // Submit and wait
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->cmdBuf
    };
    vkQueueSubmit(ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->queue);

    // Copy output
    float* dataOut = (float*)((char*)ctx->mapped + ctx->alignedSize * 2);
    memcpy(out, dataOut, n * sizeof(float));

    return 0;
}

size_t gillies_max_elements(GilliesContext* ctx) {
    return ctx ? ctx->bufferSize / sizeof(float) : 0;
}
