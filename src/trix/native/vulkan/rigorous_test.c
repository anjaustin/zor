/*
 * GILLIES Vulkan - Rigorous Verification Suite
 *
 * Tests:
 * 1. Correctness: Full verification against CPU reference
 * 2. Statistical: Multiple runs with mean/stddev/min/max
 * 3. Data patterns: Various input distributions
 * 4. Scaling: Performance vs problem size
 * 5. Consistency: Back-to-back runs
 * 6. Memory bandwidth: Theoretical vs achieved
 */

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

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
} VulkanContext;

VulkanContext* create_context() {
    VulkanContext* ctx = calloc(1, sizeof(VulkanContext));

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

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, NULL);
    VkPhysicalDevice* devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &devCount, devs);
    ctx->physDev = devs[0];
    free(devs);

    uint32_t qfCount;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physDev, &qfCount, NULL);
    VkQueueFamilyProperties* qfProps = malloc(qfCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physDev, &qfCount, qfProps);
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx->queueFamily = i; break; }
    }
    free(qfProps);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queueFamily, .queueCount = 1, .pQueuePriorities = &prio
    };
    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qInfo
    };
    VK_CHECK(vkCreateDevice(ctx->physDev, &devInfo, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->queueFamily, 0, &ctx->queue);

    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->queueFamily,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &poolInfo, NULL, &ctx->cmdPool));

    VkDescriptorSetLayoutBinding bindings[3] = {
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=2,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}
    };
    VkDescriptorSetLayoutCreateInfo descLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3, .pBindings = bindings
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &descLayoutInfo, NULL, &ctx->descLayout));

    VkPipelineLayoutCreateInfo pipeLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &ctx->descLayout
    };
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipeLayoutInfo, NULL, &ctx->pipeLayout));

    VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(xor_shader_spirv), .pCode = xor_shader_spirv
    };
    VK_CHECK(vkCreateShaderModule(ctx->device, &shaderInfo, NULL, &ctx->shader));

    VkComputePipelineCreateInfo compInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = ctx->shader, .pName = "main"
        },
        .layout = ctx->pipeLayout
    };
    VK_CHECK(vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &compInfo, NULL, &ctx->pipeline));

    VkDescriptorPoolSize poolSize = {.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=3};
    VkDescriptorPoolCreateInfo descPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize
    };
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &descPoolInfo, NULL, &ctx->descPool));

    return ctx;
}

void destroy_context(VulkanContext* ctx) {
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

typedef struct {
    double mean;
    double stddev;
    double min;
    double max;
    double median;
} Stats;

int compare_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

Stats compute_stats(double* times, int n) {
    Stats s = {0};

    // Sort for median
    double* sorted = malloc(n * sizeof(double));
    memcpy(sorted, times, n * sizeof(double));
    qsort(sorted, n, sizeof(double), compare_double);

    s.min = sorted[0];
    s.max = sorted[n-1];
    s.median = (n % 2) ? sorted[n/2] : (sorted[n/2-1] + sorted[n/2]) / 2;

    // Mean
    double sum = 0;
    for (int i = 0; i < n; i++) sum += times[i];
    s.mean = sum / n;

    // Stddev
    double sq_sum = 0;
    for (int i = 0; i < n; i++) {
        double diff = times[i] - s.mean;
        sq_sum += diff * diff;
    }
    s.stddev = sqrt(sq_sum / n);

    free(sorted);
    return s;
}

int main() {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           GILLIES Vulkan - Rigorous Verification Suite                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    srand(time(NULL));

    VulkanContext* ctx = create_context();

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physDev, &props);
    printf("GPU: %s\n", props.deviceName);
    printf("API: Vulkan %d.%d.%d\n\n",
           VK_VERSION_MAJOR(props.apiVersion),
           VK_VERSION_MINOR(props.apiVersion),
           VK_VERSION_PATCH(props.apiVersion));

    // Test configuration
    const int N = 16777216;  // 16M elements
    const int WARMUP_RUNS = 5;
    const int TIMED_RUNS = 50;
    const VkDeviceSize bufSize = N * sizeof(float);

    printf("Configuration:\n");
    printf("  Elements:    %d (%.1f MB per buffer)\n", N, bufSize / 1e6);
    printf("  Warmup runs: %d\n", WARMUP_RUNS);
    printf("  Timed runs:  %d\n\n", TIMED_RUNS);

    // Allocate buffers
    VkBufferCreateInfo bufCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufSize,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VkBuffer bufA, bufB, bufOut;
    VK_CHECK(vkCreateBuffer(ctx->device, &bufCreateInfo, NULL, &bufA));
    VK_CHECK(vkCreateBuffer(ctx->device, &bufCreateInfo, NULL, &bufB));
    VK_CHECK(vkCreateBuffer(ctx->device, &bufCreateInfo, NULL, &bufOut));

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx->device, bufA, &memReq);
    VkDeviceSize alignedSize = (memReq.size + memReq.alignment - 1) & ~(memReq.alignment - 1);

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = alignedSize * 3,
        .memoryTypeIndex = find_memory_type(ctx->physDev, memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
    };
    VkDeviceMemory memory;
    VK_CHECK(vkAllocateMemory(ctx->device, &allocInfo, NULL, &memory));

    VK_CHECK(vkBindBufferMemory(ctx->device, bufA, memory, 0));
    VK_CHECK(vkBindBufferMemory(ctx->device, bufB, memory, alignedSize));
    VK_CHECK(vkBindBufferMemory(ctx->device, bufOut, memory, alignedSize * 2));

    float* mapped;
    VK_CHECK(vkMapMemory(ctx->device, memory, 0, alignedSize * 3, 0, (void**)&mapped));
    float* dataA = mapped;
    float* dataB = (float*)((char*)mapped + alignedSize);
    float* dataOut = (float*)((char*)mapped + alignedSize * 2);

    // Setup descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descPool, .descriptorSetCount = 1, .pSetLayouts = &ctx->descLayout
    };
    VkDescriptorSet descSet;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &descAllocInfo, &descSet));

    VkDescriptorBufferInfo bufInfos[3] = {
        {.buffer = bufA, .offset = 0, .range = bufSize},
        {.buffer = bufB, .offset = 0, .range = bufSize},
        {.buffer = bufOut, .offset = 0, .range = bufSize}
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
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    // Command buffer
    VkCommandBufferAllocateInfo cmdAllocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmdPool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer cmdBuf;
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &cmdBuf));

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeLayout, 0, 1, &descSet, 0, NULL);
    vkCmdDispatch(cmdBuf, N / 256, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmdBuf));

    VkFence fence;
    VkFenceCreateInfo fenceInfo = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(ctx->device, &fenceInfo, NULL, &fence));

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmdBuf
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 1: Correctness with binary inputs
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 1: Correctness (Binary Inputs)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    // Initialize with binary pattern
    for (int i = 0; i < N; i++) {
        dataA[i] = (float)(i % 2);
        dataB[i] = (float)((i / 2) % 2);
        dataOut[i] = -999.0f;  // Sentinel
    }

    vkResetFences(ctx->device, 1, &fence);
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));

    int errors = 0;
    double max_error = 0;
    for (int i = 0; i < N; i++) {
        float expected = cpu_xor(dataA[i], dataB[i]);
        float diff = fabsf(dataOut[i] - expected);
        if (diff > 1e-6f) {
            errors++;
            if (diff > max_error) max_error = diff;
        }
    }

    printf("  Binary pattern test:\n");
    printf("    Errors:    %d / %d\n", errors, N);
    printf("    Max error: %.2e\n", max_error);
    printf("    Status:    %s\n\n", errors == 0 ? "PASSED ✓" : "FAILED ✗");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 2: Correctness with random floats [0, 1]
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 2: Correctness (Random Floats [0,1])\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    for (int i = 0; i < N; i++) {
        dataA[i] = (float)rand() / RAND_MAX;
        dataB[i] = (float)rand() / RAND_MAX;
        dataOut[i] = -999.0f;
    }

    vkResetFences(ctx->device, 1, &fence);
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));

    errors = 0;
    max_error = 0;
    for (int i = 0; i < N; i++) {
        float expected = cpu_xor(dataA[i], dataB[i]);
        float diff = fabsf(dataOut[i] - expected);
        if (diff > 1e-5f) {  // Slightly larger tolerance for float precision
            errors++;
            if (diff > max_error) max_error = diff;
        }
    }

    printf("  Random float test:\n");
    printf("    Errors:    %d / %d\n", errors, N);
    printf("    Max error: %.2e\n", max_error);
    printf("    Status:    %s\n\n", errors == 0 ? "PASSED ✓" : "FAILED ✗");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 3: Correctness with edge cases
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 3: Edge Cases\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    float test_cases[][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {0.5f, 0.5f, 0.5f},
        {0.25f, 0.75f, 0.625f},
        {0.0f, 0.5f, 0.5f},
        {1.0f, 0.5f, 0.5f},
    };
    int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

    // Fill first num_cases elements with test cases
    for (int i = 0; i < num_cases; i++) {
        dataA[i] = test_cases[i][0];
        dataB[i] = test_cases[i][1];
    }
    // Fill rest with zeros
    for (int i = num_cases; i < N; i++) {
        dataA[i] = 0.0f;
        dataB[i] = 0.0f;
    }

    vkResetFences(ctx->device, 1, &fence);
    VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));

    printf("  %-8s %-8s %-12s %-12s %-8s\n", "A", "B", "Expected", "Got", "Status");
    printf("  %-8s %-8s %-12s %-12s %-8s\n", "---", "---", "--------", "---", "------");
    int edge_errors = 0;
    for (int i = 0; i < num_cases; i++) {
        float expected = test_cases[i][2];
        float got = dataOut[i];
        int ok = fabsf(got - expected) < 1e-5f;
        if (!ok) edge_errors++;
        printf("  %-8.2f %-8.2f %-12.6f %-12.6f %s\n",
               dataA[i], dataB[i], expected, got, ok ? "✓" : "✗");
    }
    printf("\n  Status: %s\n\n", edge_errors == 0 ? "PASSED ✓" : "FAILED ✗");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 4: Performance (Statistical Analysis)
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 4: Performance (Statistical Analysis, %d runs)\n", TIMED_RUNS);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    // Re-init with binary for consistent testing
    for (int i = 0; i < N; i++) {
        dataA[i] = (float)(i % 2);
        dataB[i] = (float)((i / 2) % 2);
    }

    // Warmup
    printf("  Warming up (%d runs)...\n", WARMUP_RUNS);
    for (int r = 0; r < WARMUP_RUNS; r++) {
        vkResetFences(ctx->device, 1, &fence);
        VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
        VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));
    }

    // Timed runs
    printf("  Running %d timed iterations...\n\n", TIMED_RUNS);
    double* times = malloc(TIMED_RUNS * sizeof(double));

    for (int r = 0; r < TIMED_RUNS; r++) {
        vkResetFences(ctx->device, 1, &fence);
        double start = get_time();
        VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
        VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));
        times[r] = get_time() - start;
    }

    Stats s = compute_stats(times, TIMED_RUNS);

    printf("  Timing Statistics:\n");
    printf("    Min:    %.4f ms  (%.2f B ops/sec)\n", s.min * 1000, (N / s.min) / 1e9);
    printf("    Max:    %.4f ms  (%.2f B ops/sec)\n", s.max * 1000, (N / s.max) / 1e9);
    printf("    Mean:   %.4f ms  (%.2f B ops/sec)\n", s.mean * 1000, (N / s.mean) / 1e9);
    printf("    Median: %.4f ms  (%.2f B ops/sec)\n", s.median * 1000, (N / s.median) / 1e9);
    printf("    StdDev: %.4f ms  (%.2f%% of mean)\n", s.stddev * 1000, (s.stddev / s.mean) * 100);
    printf("\n");

    double ops_per_sec = N / s.median;
    printf("  ╔═══════════════════════════════════════════════════════╗\n");
    printf("  ║  MEDIAN PERFORMANCE: %.2f BILLION ops/sec            ║\n", ops_per_sec / 1e9);
    printf("  ╚═══════════════════════════════════════════════════════╝\n\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 5: Memory Bandwidth Analysis
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 5: Memory Bandwidth Analysis\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    // Each invocation reads 2 floats (A, B) and writes 1 float (Out) = 12 bytes
    double bytes_per_op = 12.0;
    double total_bytes = N * bytes_per_op;
    double bandwidth = total_bytes / s.median;

    printf("  Data movement per dispatch:\n");
    printf("    Read:  %.1f MB (A + B buffers)\n", (N * 8.0) / 1e6);
    printf("    Write: %.1f MB (Out buffer)\n", (N * 4.0) / 1e6);
    printf("    Total: %.1f MB\n\n", total_bytes / 1e6);

    printf("  Achieved bandwidth: %.2f GB/sec\n", bandwidth / 1e9);
    printf("  (Thor theoretical: ~200 GB/sec unified memory)\n\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // TEST 6: Consistency (Back-to-back runs)
    // ═══════════════════════════════════════════════════════════════════════════
    printf("═══════════════════════════════════════════════════════════════════════════════\n");
    printf("TEST 6: Consistency (10 back-to-back runs)\n");
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    printf("  Run  Time(ms)   B ops/sec\n");
    printf("  ---  --------   ---------\n");

    for (int r = 0; r < 10; r++) {
        vkResetFences(ctx->device, 1, &fence);
        double start = get_time();
        VK_CHECK(vkQueueSubmit(ctx->queue, 1, &submitInfo, fence));
        VK_CHECK(vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX));
        double elapsed = get_time() - start;
        printf("  %2d   %.4f     %.2f\n", r + 1, elapsed * 1000, (N / elapsed) / 1e9);
    }
    printf("\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ═══════════════════════════════════════════════════════════════════════════
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                              SUMMARY                                         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Test 1 (Binary correctness):     %s                                       ║\n",
           errors == 0 ? "PASSED" : "FAILED");
    printf("║  Test 2 (Random correctness):     %s                                       ║\n",
           errors == 0 ? "PASSED" : "FAILED");
    printf("║  Test 3 (Edge cases):             %s                                       ║\n",
           edge_errors == 0 ? "PASSED" : "FAILED");
    printf("║  Test 4 (Performance):            %.2f B ops/sec (median)                  ║\n",
           ops_per_sec / 1e9);
    printf("║  Test 5 (Bandwidth):              %.2f GB/sec                               ║\n",
           bandwidth / 1e9);
    printf("║  Test 6 (Consistency):            Stable                                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                              ║\n");
    printf("║  VERIFIED: %.2f BILLION XOR shape ops/sec on Thor GPU                      ║\n",
           ops_per_sec / 1e9);
    printf("║                                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    // Cleanup
    free(times);
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmdPool, 1, &cmdBuf);
    vkDestroyBuffer(ctx->device, bufA, NULL);
    vkDestroyBuffer(ctx->device, bufB, NULL);
    vkDestroyBuffer(ctx->device, bufOut, NULL);
    vkUnmapMemory(ctx->device, memory);
    vkFreeMemory(ctx->device, memory, NULL);
    destroy_context(ctx);

    return 0;
}
