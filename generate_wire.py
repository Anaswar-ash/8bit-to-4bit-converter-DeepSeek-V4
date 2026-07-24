code = """
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

static void *qalloc(size_t n){
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(n, 4096);
#else
    if(posix_memalign(&p, 4096, n)) p = NULL;
#endif
    if(!p){ fprintf(stderr, "OOM qalloc %zu\\n", n); exit(1); }
    return p;
}

static void qt_load(shards *S, QT *t, const char *name) {
    if (!st_has(S, name)) { memset(t, 0, sizeof(QT)); return; }
    
    char buf[256]; snprintf(buf, sizeof(buf), "%s.qs", name);
    if (st_has(S, buf)) { // It's a quantized tensor (INT4)
        t->fmt = 2;
        int64_t shape_0 = 1, shape_1 = 1;
        // In colibri we don't store shape explicitly in QT, but we can compute it from numel.
        // wait, st_tensor has shape[4] and ndim! We can get it via st_find!
        st_tensor *st = st_find(S, name);
        t->O = st->shape[0];
        if (st->ndim > 1) t->I = st->shape[1] * 2;
        else t->I = st->shape[0] * 2;
        
        t->q4 = qalloc(st->nbytes);
        st_read_raw(S, name, t->q4, 0);
        
        t->s = qalloc(t->O * sizeof(float));
        st_read_f32(S, buf, t->s, 0);
    } else { // F32
        st_tensor *st = st_find(S, name);
        t->fmt = 0;
        t->O = st->shape[0];
        if (st->ndim > 1) t->I = st->shape[1];
        else t->I = 1;
        
        t->qf = qalloc(st->numel * sizeof(float));
        st_read_f32(S, name, t->qf, 0);
    }
}

static float* fp32_load(shards *S, const char *name) {
    if (!st_has(S, name)) return NULL;
    int64_t numel = st_numel(S, name);
    float *p = qalloc(numel * sizeof(float));
    st_read_f32(S, name, p, 0);
    return p;
}

static void wire_tensors(Model *m) {
    shards *S = &m->S;
    qt_load(S, &m->embed, "embed.weight");
    qt_load(S, &m->lm_head, "lm_head.weight");
    m->final_norm = fp32_load(S, "norm.weight");
    
    m->hc_head_fn = fp32_load(S, "hc_head_fn");
    m->hc_head_base = fp32_load(S, "hc_head_base");
    m->hc_head_scale = fp32_load(S, "hc_head_scale");
    
    char buf[256];
    for (int i = 0; i < m->c.n_layers; i++) {
        Layer *l = &m->L[i];
        l->ffn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.ffn_norm.weight", i) ? buf : buf);
        
        qt_load(S, &l->shared_w1, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w1.weight", i) ? buf : buf);
        qt_load(S, &l->shared_w2, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w2.weight", i) ? buf : buf);
        qt_load(S, &l->shared_w3, snprintf(buf, sizeof(buf), "layers.%d.ffn.shared_experts.w3.weight", i) ? buf : buf);
        
        qt_load(S, &l->gate, snprintf(buf, sizeof(buf), "layers.%d.ffn.gate.weight", i) ? buf : buf);
        l->gate_bias = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.ffn.gate.bias", i) ? buf : buf);
        
        // tid2eid needs to be cast to int*, but since fp32_load returns float* (and st_read_f32 does fp32 conversion),
        // we might have a problem if it's float. In safetensors it's float32? We will just leave it as float* in Layer struct!
        
        l->attn_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn_norm.weight", i) ? buf : buf);
        qt_load(S, &l->wq_a, snprintf(buf, sizeof(buf), "layers.%d.attn.wq_a.weight", i) ? buf : buf);
        qt_load(S, &l->wq_b, snprintf(buf, sizeof(buf), "layers.%d.attn.wq_b.weight", i) ? buf : buf);
        l->q_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn.q_norm.weight", i) ? buf : buf);
        
        qt_load(S, &l->wkv, snprintf(buf, sizeof(buf), "layers.%d.attn.wkv.weight", i) ? buf : buf);
        l->kv_norm = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.attn.kv_norm.weight", i) ? buf : buf);
        
        qt_load(S, &l->wo_a, snprintf(buf, sizeof(buf), "layers.%d.attn.wo_a.weight", i) ? buf : buf);
        qt_load(S, &l->wo_b, snprintf(buf, sizeof(buf), "layers.%d.attn.wo_b.weight", i) ? buf : buf);
        
        l->hc_attn_base = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_base", i) ? buf : buf);
        l->hc_attn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_scale", i) ? buf : buf);
        qt_load(S, &l->hc_attn_fn, snprintf(buf, sizeof(buf), "layers.%d.hc_attn_fn", i) ? buf : buf);
        
        l->hc_ffn_base = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_base", i) ? buf : buf);
        l->hc_ffn_scale = fp32_load(S, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_scale", i) ? buf : buf);
        qt_load(S, &l->hc_ffn_fn, snprintf(buf, sizeof(buf), "layers.%d.hc_ffn_fn", i) ? buf : buf);
        
        // Experts are NOT loaded here to prevent OOM! 
        // We will stream them in the forward pass!
    }
}
"""
open('../colibri-main/colibri-main/c/generate_wire.py', 'w').write(code)
