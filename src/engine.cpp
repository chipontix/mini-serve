// The model runner: a Llama-architecture forward pass over a paged KV cache.
//
// Matrix multiplications run on the CPU through Accelerate (cblas_sgemm); attention runs on the
// GPU through kvb::run_paged, the paged decode-attention kernel of kv-cache-bench. Python owns
// everything else (block allocation, scheduling, sampling) and calls this through a C API with
// ctypes (mini_serve/engine.py).
//
// One forward call takes a flat list of tokens from any number of sequences. Token i sits at
// position positions[i] of its sequence and comes with that sequence's block table (one row of
// block_tables). The call writes the token's K and V into its slot of the pool, then lets the
// token attend to positions 0..positions[i] of its sequence. A prefill of an L-token prompt is
// L such rows with the same block table and positions 0..L-1; a decode step is one row per
// sequence. The kernel was written for decode (one query per sequence), and this is what makes
// it serve prefill too: every prompt token is a "sequence" of length pos+1, which is exactly a
// causal mask.

#include <Accelerate/Accelerate.h>

#include <algorithm>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "metal_attn.hpp"

namespace {

// Must match class Config in mini_serve/engine.py.
struct Config {
    int n_layers, hidden, n_heads, n_kv_heads, head_dim, inter, vocab;
    float rope_theta, rms_eps;
    int num_blocks, page_size, kv_f16;
};

struct Layer {
    std::vector<float> attn_norm, wqkv, wo, mlp_norm, wgu, wdown;
    std::unique_ptr<kvb::Buffer> k, v;  // [num_blocks][Hkv][page][D], fp16 or fp32
};

// y[n][out] = x[n][in] * W[out][in]^T + beta * y
void matmul(int n, int out, int in, const float* x, const float* w, float* y, float beta = 0.0f) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out, in, 1.0f, x, in, w, in, beta, y, out);
}

void rmsnorm(const float* x, const float* w, float* y, int dim, float eps) {
    double ss = 0;  // HF accumulates in fp32; fp64 here only removes summation-order noise
    for (int i = 0; i < dim; ++i) ss += double(x[i]) * x[i];
    const float r = 1.0f / std::sqrt(float(ss / dim) + eps);
    for (int i = 0; i < dim; ++i) y[i] = w[i] * (x[i] * r);
}

void grow(std::unique_ptr<kvb::Buffer>& b, size_t bytes) {
    if (!b || b->size() < bytes) b = std::make_unique<kvb::Buffer>(bytes);
}

struct Engine {
    Config c;
    std::vector<float> embed, final_norm, inv_freq;
    std::vector<Layer> layers;
    int loaded = 0;
    std::unique_ptr<kvb::Buffer> q, out, table, lens;  // per-step buffers shared with the GPU
    std::vector<float> x, h, qkv, gu, act;

    explicit Engine(const Config& cfg) : c(cfg) {
        if (c.head_dim != 64 && c.head_dim != 128) throw std::invalid_argument("head_dim must be 64 or 128");
        if (c.n_heads % c.n_kv_heads) throw std::invalid_argument("n_heads % n_kv_heads != 0");
        if (c.num_blocks <= 0 || c.page_size <= 0) throw std::invalid_argument("bad pool size");
        if (!kvb::metal_available()) throw std::runtime_error("no Metal device");
        const size_t pool = size_t(c.num_blocks) * c.n_kv_heads * c.page_size * c.head_dim * (c.kv_f16 ? 2 : 4);
        layers.resize(size_t(c.n_layers));
        for (Layer& l : layers) {
            l.k = std::make_unique<kvb::Buffer>(pool);
            l.v = std::make_unique<kvb::Buffer>(pool);
        }
        // Same arithmetic as transformers' default RoPE: fp32 inverse frequencies, fp32 angles.
        for (int i = 0; i < c.head_dim / 2; ++i)
            inv_freq.push_back(1.0f / std::pow(c.rope_theta, float(2 * i) / float(c.head_dim)));
    }

    int qkv_dim() const { return (c.n_heads + 2 * c.n_kv_heads) * c.head_dim; }
    int expected_weights() const { return 2 + 9 * c.n_layers; }

    // Takes a tensor by its Hugging Face name. Q/K/V are stacked into one matrix and gate/up
    // into another, so each needs one GEMM instead of three or two.
    void set_weight(const std::string& name, const float* data, int64_t n) {
        const int D = c.head_dim, H = c.n_heads, Hkv = c.n_kv_heads, E = c.hidden;
        auto put = [&](std::vector<float>& dst, size_t total, size_t offset, size_t count) {
            if (size_t(n) != count) throw std::invalid_argument(name + ": wrong size " + std::to_string(n));
            dst.resize(total);
            std::memcpy(dst.data() + offset, data, count * sizeof(float));
            ++loaded;
        };
        if (name == "model.embed_tokens.weight") return put(embed, size_t(c.vocab) * E, 0, size_t(c.vocab) * E);
        if (name == "model.norm.weight") return put(final_norm, E, 0, E);
        int i = -1;
        char rest[128] = {};
        if (std::sscanf(name.c_str(), "model.layers.%d.%127s", &i, rest) != 2 || i < 0 || i >= c.n_layers)
            throw std::invalid_argument("unknown weight " + name);
        Layer& l = layers[size_t(i)];
        const std::string r = rest;
        const size_t qkv = size_t(qkv_dim()) * E, gu = size_t(2) * c.inter * E;
        if (r == "input_layernorm.weight") return put(l.attn_norm, E, 0, E);
        if (r == "post_attention_layernorm.weight") return put(l.mlp_norm, E, 0, E);
        if (r == "self_attn.q_proj.weight") return put(l.wqkv, qkv, 0, size_t(H) * D * E);
        if (r == "self_attn.k_proj.weight") return put(l.wqkv, qkv, size_t(H) * D * E, size_t(Hkv) * D * E);
        if (r == "self_attn.v_proj.weight") return put(l.wqkv, qkv, size_t(H + Hkv) * D * E, size_t(Hkv) * D * E);
        if (r == "self_attn.o_proj.weight") return put(l.wo, size_t(E) * H * D, 0, size_t(E) * H * D);
        if (r == "mlp.gate_proj.weight") return put(l.wgu, gu, 0, size_t(c.inter) * E);
        if (r == "mlp.up_proj.weight") return put(l.wgu, gu, size_t(c.inter) * E, size_t(c.inter) * E);
        if (r == "mlp.down_proj.weight") return put(l.wdown, size_t(E) * c.inter, 0, size_t(E) * c.inter);
        throw std::invalid_argument("unknown weight " + name);
    }

    // Rotates pairs (i, i + D/2) of every head in `v` (the non-interleaved Llama layout).
    void rope(float* v, int heads, int pos) const {
        const int half = c.head_dim / 2;
        for (int i = 0; i < half; ++i) {
            const float a = float(pos) * inv_freq[size_t(i)], cs = std::cos(a), sn = std::sin(a);
            for (int hd = 0; hd < heads; ++hd) {
                float* p = v + size_t(hd) * c.head_dim;
                const float x0 = p[i], x1 = p[i + half];
                p[i] = x0 * cs - x1 * sn;
                p[i + half] = x1 * cs + x0 * sn;
            }
        }
    }

    template <class T>
    void store_kv(const Layer& l, int n, const int32_t* positions, const int32_t* tables, int max_blocks) {
        const int D = c.head_dim, H = c.n_heads, Hkv = c.n_kv_heads, P = c.page_size;
        T* kp = l.k->as<T>();
        T* vp = l.v->as<T>();
        for (int t = 0; t < n; ++t) {
            const int pos = positions[t];
            const size_t blk = size_t(tables[size_t(t) * max_blocks + pos / P]);
            const float* src = qkv.data() + size_t(t) * qkv_dim();
            for (int kvh = 0; kvh < Hkv; ++kvh) {
                const size_t dst = ((blk * Hkv + kvh) * P + pos % P) * D;
                for (int d = 0; d < D; ++d) {
                    kp[dst + d] = T(src[size_t(H + kvh) * D + d]);
                    vp[dst + d] = T(src[size_t(H + Hkv + kvh) * D + d]);
                }
            }
        }
    }

    void forward(int n, const int32_t* tokens, const int32_t* positions, const int32_t* tables,
                 int max_blocks, int n_out, const int32_t* out_rows, float* logits) {
        if (loaded != expected_weights()) throw std::runtime_error("not all weights are loaded");
        if (n <= 0 || max_blocks <= 0 || n_out < 0) throw std::invalid_argument("bad sizes");
        const int D = c.head_dim, H = c.n_heads, E = c.hidden, P = c.page_size, QKV = qkv_dim();
        int max_len = 0;
        for (int t = 0; t < n; ++t) {
            const int pos = positions[t];
            if (tokens[t] < 0 || tokens[t] >= c.vocab) throw std::invalid_argument("token id out of range");
            if (pos < 0 || pos / P >= max_blocks) throw std::invalid_argument("position outside the block table");
            for (int b = 0; b <= pos / P; ++b) {
                const int blk = tables[size_t(t) * max_blocks + b];
                if (blk < 0 || blk >= c.num_blocks) throw std::invalid_argument("block id out of range");
            }
            max_len = std::max(max_len, pos + 1);
        }
        for (int r = 0; r < n_out; ++r)
            if (out_rows[r] < 0 || out_rows[r] >= n) throw std::invalid_argument("out_rows out of range");

        grow(q, size_t(n) * H * D * sizeof(float));
        grow(out, size_t(n) * H * D * sizeof(float));
        grow(table, size_t(n) * max_blocks * sizeof(int32_t));
        grow(lens, size_t(n) * sizeof(int32_t));
        std::memcpy(table->data(), tables, size_t(n) * max_blocks * sizeof(int32_t));
        for (int t = 0; t < n; ++t) lens->as<int32_t>()[t] = positions[t] + 1;
        x.resize(size_t(n) * E);
        h.resize(size_t(n) * E);
        qkv.resize(size_t(n) * QKV);
        gu.resize(size_t(n) * 2 * c.inter);
        act.resize(size_t(n) * c.inter);
        for (int t = 0; t < n; ++t)
            std::memcpy(&x[size_t(t) * E], &embed[size_t(tokens[t]) * E], size_t(E) * sizeof(float));

        kvb::PagedArgs pa;
        pa.shape = kvb::Shape{n, H, c.n_kv_heads, D};
        pa.type = c.kv_f16 ? kvb::KvType::f16 : kvb::KvType::f32;
        pa.page_size = P;
        pa.max_blocks = max_blocks;
        pa.max_len = max_len;
        pa.q = q.get();
        pa.block_table = table.get();
        pa.seq_lens = lens.get();
        pa.out = out.get();

        for (const Layer& l : layers) {
            for (int t = 0; t < n; ++t) rmsnorm(&x[size_t(t) * E], l.attn_norm.data(), &h[size_t(t) * E], E, c.rms_eps);
            matmul(n, QKV, E, h.data(), l.wqkv.data(), qkv.data());
            for (int t = 0; t < n; ++t) {
                float* row = &qkv[size_t(t) * QKV];
                rope(row, H + c.n_kv_heads, positions[t]);  // q heads, then k heads, are adjacent
                std::memcpy(q->as<float>() + size_t(t) * H * D, row, size_t(H) * D * sizeof(float));
            }
            if (c.kv_f16) store_kv<kvb::half_t>(l, n, positions, tables, max_blocks);
            else store_kv<float>(l, n, positions, tables, max_blocks);
            pa.k = l.k.get();
            pa.v = l.v.get();
            kvb::run_paged(pa);  // waits for the GPU; out = [n][H][D]
            matmul(n, E, H * D, out->as<float>(), l.wo.data(), x.data(), 1.0f);  // x += attn · Wo^T

            for (int t = 0; t < n; ++t) rmsnorm(&x[size_t(t) * E], l.mlp_norm.data(), &h[size_t(t) * E], E, c.rms_eps);
            matmul(n, 2 * c.inter, E, h.data(), l.wgu.data(), gu.data());
            for (int t = 0; t < n; ++t)
                for (int i = 0; i < c.inter; ++i) {
                    const float g = gu[size_t(t) * 2 * c.inter + i], u = gu[size_t(t) * 2 * c.inter + c.inter + i];
                    act[size_t(t) * c.inter + i] = g / (1.0f + std::exp(-g)) * u;  // SiLU(gate) * up
                }
            matmul(n, E, c.inter, act.data(), l.wdown.data(), x.data(), 1.0f);  // x += mlp
        }
        if (n_out == 0) return;
        for (int r = 0; r < n_out; ++r)
            rmsnorm(&x[size_t(out_rows[r]) * E], final_norm.data(), &h[size_t(r) * E], E, c.rms_eps);
        matmul(n_out, c.vocab, E, h.data(), embed.data(), logits);  // tied embeddings: lm_head = embed
    }
};

thread_local std::string g_error;

template <class F>
int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // namespace

extern "C" {

const char* ms_last_error() { return g_error.c_str(); }

void* ms_create(const Config* cfg) {
    Engine* e = nullptr;
    guarded([&] { e = new Engine(*cfg); });
    return e;
}

void ms_destroy(void* e) { delete static_cast<Engine*>(e); }

int ms_set_weight(void* e, const char* name, const float* data, int64_t n) {
    return guarded([&] { static_cast<Engine*>(e)->set_weight(name, data, n); });
}

// tokens, positions: [n]; block_tables: [n][max_blocks]; out_rows: [n_out] indices into the n
// tokens whose logits are wanted; logits: [n_out][vocab]. Returns 0, or -1 (see ms_last_error).
int ms_forward(void* e, int n, const int32_t* tokens, const int32_t* positions, const int32_t* block_tables,
               int max_blocks, int n_out, const int32_t* out_rows, float* logits) {
    return guarded([&] {
        static_cast<Engine*>(e)->forward(n, tokens, positions, block_tables, max_blocks, n_out, out_rows, logits);
    });
}

}  // extern "C"
