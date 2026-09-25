#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <ze_api.h>
#include <ze_graph_ext.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ppocr {

struct TensorDesc {
    std::string name;
    std::vector<uint32_t> dims;
    size_t total_bytes = 0;
    bool is_input = true;
};

// Represents a single asynchronous inference request slot with its own command list, USM buffers, and sync event
class NpuInferRequest {
public:
    NpuInferRequest(
        ze_context_handle_t context,
        ze_device_handle_t device,
        ze_command_queue_handle_t queue,
        uint32_t queue_ordinal,
        ze_graph_handle_t graph,
        const std::vector<TensorDesc>& inputs,
        const std::vector<TensorDesc>& outputs,
        ze_graph_dditable_ext_t* graph_api
    );
    ~NpuInferRequest();

    void* get_input_ptr(size_t index = 0) const { return input_ptrs_[index]; }
    void* get_output_ptr(size_t index = 0) const { return output_ptrs_[index]; }
    size_t get_input_size(size_t index = 0) const { return input_descs_[index].total_bytes; }
    size_t get_output_size(size_t index = 0) const { return output_descs_[index].total_bytes; }

    // Submits the inference task asynchronously to NPU command queue
    void start_async();

    // Waits for NPU execution to complete (blocking)
    void wait();

private:
    ze_context_handle_t context_;
    ze_device_handle_t device_;
    ze_command_queue_handle_t queue_;
    ze_graph_handle_t graph_;
    ze_graph_dditable_ext_t* graph_api_;

    std::vector<TensorDesc> input_descs_;
    std::vector<TensorDesc> output_descs_;

    std::vector<void*> input_ptrs_;
    std::vector<void*> output_ptrs_;

    ze_command_list_handle_t command_list_ = nullptr;
};

// Manages a compiled model instance on NPU
class NpuGraph {
public:
    NpuGraph(
        ze_context_handle_t context,
        ze_device_handle_t device,
        ze_command_queue_handle_t queue,
        uint32_t queue_ordinal,
        const std::string& model_path,
        ze_graph_dditable_ext_t* graph_api,
        size_t num_requests = 2 // Ping-Pong double buffering
    );
    ~NpuGraph();

    // Acquires an available infer request from the pool
    NpuInferRequest& get_request(size_t slot = 0) { return *requests_[slot % requests_.size()]; }
    size_t request_count() const { return requests_.size(); }

    const std::vector<TensorDesc>& inputs() const { return inputs_; }
    const std::vector<TensorDesc>& outputs() const { return outputs_; }

private:
    void init_graph(const std::string& model_path);
    void query_arguments();

    ze_context_handle_t context_;
    ze_device_handle_t device_;
    ze_command_queue_handle_t queue_;
    uint32_t queue_ordinal_;
    ze_graph_dditable_ext_t* graph_api_;
    ze_graph_handle_t graph_ = nullptr;

    std::vector<TensorDesc> inputs_;
    std::vector<TensorDesc> outputs_;
    std::vector<std::unique_ptr<NpuInferRequest>> requests_;
};

// Singleton context managing the Intel NPU Level Zero driver
class NpuEngine {
public:
    static NpuEngine& instance();

    bool initialize();

    std::shared_ptr<NpuGraph> load_model(const std::string& model_path, size_t num_requests = 2);

    ze_context_handle_t context() const { return context_; }
    ze_device_handle_t device() const { return device_; }
    ze_command_queue_handle_t queue() const { return queue_; }
    ze_graph_dditable_ext_t* graph_api() const { return graph_api_; }

    const std::string& device_name() const { return device_name_; }

private:
    NpuEngine() = default;
    ~NpuEngine();

    bool initialized_ = false;
    ze_driver_handle_t driver_ = nullptr;
    ze_device_handle_t device_ = nullptr;
    ze_context_handle_t context_ = nullptr;
    ze_command_queue_handle_t queue_ = nullptr;
    uint32_t queue_ordinal_ = 0;
    ze_graph_dditable_ext_t* graph_api_ = nullptr;
    std::string device_name_ = "Unknown NPU";
};

} // namespace ppocr
