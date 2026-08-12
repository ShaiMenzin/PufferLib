#define PRECISION_FLOAT
#include "../src/models.cu"

extern "C" void mingru_recurrent_scan(
        void* combined, void* state, void* input, void* episode_starts,
        void* out, void* next_state, void* a_star, void* s_vals, void* log_values,
        void* grad_out, void* grad_next_state, void* grad_combined,
        void* grad_state, void* grad_input, int B, int T, int H) {
    PrefixScan scan = {};
    scan.combined_ptr = (float*)combined;
    scan.state_ptr = (float*)state;
    scan.input_ptr = (float*)input;
    scan.episode_starts_ptr = (float*)episode_starts;
    scan.B = B;
    scan.T = T;
    scan.H = H;
    scan.a_star = {.data = (float*)a_star, .shape = {B, T + 1, H}};
    scan.s_vals = {.data = (float*)s_vals, .shape = {B, T + 1, H}};
    scan.log_values_buf = {.data = (float*)log_values, .shape = {B, T + 1, H}};
    scan.out = {.data = (float*)out, .shape = {B, T, H}};
    scan.next_state = {.data = (float*)next_state, .shape = {B, 1, H}};
    scan.grad_combined = {.data = (float*)grad_combined, .shape = {B, T, 3 * H}};
    scan.grad_state = {.data = (float*)grad_state, .shape = {B, 1, H}};
    scan.grad_input = {.data = (float*)grad_input, .shape = {B, T, H}};

    mingru_scan_forward<<<grid_size(B * H), BLOCK_SIZE>>>(scan);
    mingru_scan_backward<<<grid_size(B * H), BLOCK_SIZE>>>(
        scan,
        (float*)grad_out,
        (float*)grad_next_state
    );
    cudaDeviceSynchronize();
}
