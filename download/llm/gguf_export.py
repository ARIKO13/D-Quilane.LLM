"""
GGUF Exporter for D'QUILANE Blueprint v2
========================================
Converts model.json to GGUF binary format.

GGUF format (custom, compatible with llama.cpp structure):
  - Magic: "GGUF" (0x46554747)
  - Version: 1 (uint32)
  - Tensor count (uint64)
  - Metadata KV count (uint64)
  - Metadata (key-value pairs)
  - Tensor info (name, dims, type, offset)
  - Tensor data (F32, alignment-padded)

Model architecture: Blueprint v2 (LFM2 hybrid + ReXMoE + XNC + AION)
All weights stored as F32 (no quantization, per user request).
"""
import json
import struct
import os
import sys
from pathlib import Path

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 1
GGUF_ALIGNMENT = 32  # bytes alignment for tensor data

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_UINT64 = 6
GGUF_TYPE_INT64 = 7
GGUF_TYPE_FLOAT32 = 8
GGUF_TYPE_BOOL = 9
GGUF_TYPE_STRING = 10
GGUF_TYPE_ARRAY = 11

# GGUF tensor types
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1


def write_string(f, s):
    """Write string: length (uint64) + bytes (UTF-8)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_value(f, value, vtype):
    """Write a metadata value based on type."""
    if vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack('<I', int(value)))
    elif vtype == GGUF_TYPE_INT32:
        f.write(struct.pack('<i', int(value)))
    elif vtype == GGUF_TYPE_UINT64:
        f.write(struct.pack('<Q', int(value)))
    elif vtype == GGUF_TYPE_INT64:
        f.write(struct.pack('<q', int(value)))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack('<f', float(value)))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(struct.pack('<B', 1 if value else 0))
    elif vtype == GGUF_TYPE_STRING:
        write_string(f, str(value))
    else:
        raise ValueError(f"Unsupported type: {vtype}")


def write_metadata_kv(f, key, value, vtype):
    """Write key-value metadata pair."""
    write_string(f, key)
    f.write(struct.pack('<I', vtype))
    write_value(f, value, vtype)


def export_gguf(model_json_path, gguf_path):
    """Convert model.json to GGUF binary format."""
    print(f"Loading model: {model_json_path}")
    with open(model_json_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    config = model['config']
    stoi = model['stoi']
    itos = model['itos']
    eos_id = model.get('eos_id', config.get('vocab_size', 0) - 1)
    aion_engram = model.get('aion_engram', {})
    weights = model['weights']
    
    print(f"Architecture: {model.get('architecture', 'unknown')}")
    print(f"Techniques: {model.get('techniques', [])}")
    print(f"Vocab: {config['vocab_size']}")
    print(f"d_model: {config['d_model']}")
    print(f"n_layers: {config.get('n_layers', 'N/A')}")
    print(f"AION engram entries: {len(aion_engram)}")
    
    # Collect all tensors to write
    tensors = []  # [(name, dims_list, flat_data_float32)]
    
    def add_tensor(name, data, dims):
        """Add tensor to list. data can be nested list or flat list."""
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                # Nested - flatten
                import numpy as np
                arr = np.array(data, dtype=np.float32)
                flat = arr.flatten().tolist()
                actual_dims = list(arr.shape)
            else:
                flat = [float(x) for x in data]
                actual_dims = dims
        else:
            flat = [float(data)]
            actual_dims = [1]
        tensors.append((name, actual_dims, flat))
    
    # Add embedding weights
    if 'tok_emb_weight' in weights:
        add_tensor('tok_emb.weight', weights['tok_emb_weight'], 
                   [config['vocab_size'], config['d_model']])
    if 'pos_emb_weight' in weights:
        add_tensor('pos_emb.weight', weights['pos_emb_weight'],
                   [config['max_seq'], config['d_model']])
    if 'norm_f_weight' in weights:
        add_tensor('norm_f.weight', weights['norm_f_weight'],
                   [config['d_model']])
    
    # Check architecture type
    if 'experts_gate_weights' in weights:
        # Blueprint v2 architecture (LFM2 + ReXMoE + AION)
        print("Detected: Blueprint v2 architecture (LFM2 hybrid + ReXMoE + AION)")
        
        # ReXMoE experts (4 shared)
        n_experts = len(weights['experts_gate_weights'])
        for i in range(n_experts):
            add_tensor(f'experts.{i}.gate.weight', weights['experts_gate_weights'][i],
                       [config['d_ff'], config['d_model']])
            add_tensor(f'experts.{i}.val.weight', weights['experts_val_weights'][i],
                       [config['d_ff'], config['d_model']])
            add_tensor(f'experts.{i}.down.weight', weights['experts_down_weights'][i],
                       [config['d_model'], config['d_ff']])
        
        # Router
        add_tensor('router.gate.weight', weights['router_weight'],
                   [config['n_experts'], config['d_model']])
        
        # Conv blocks (LFM2)
        conv_blocks = weights.get('conv_blocks', [])
        for i, blk in enumerate(conv_blocks):
            add_tensor(f'conv_blocks.{i}.norm1.weight', blk['norm1_weight'], [config['d_model']])
            # conv_weight shape: [out_channels=2*d, in_channels=d, kernel] for depthwise
            # Actually for Conv1d groups=d, weight shape is [2*d, 1, kernel]
            add_tensor(f'conv_blocks.{i}.conv.weight', blk['conv_weight'], 
                       [2*config['d_model'], 1, config.get('conv_kernel', 4)])
            if blk.get('conv_bias') is not None:
                add_tensor(f'conv_blocks.{i}.conv.bias', blk['conv_bias'], [2*config['d_model']])
            add_tensor(f'conv_blocks.{i}.proj.weight', blk['proj_weight'],
                       [config['d_model'], config['d_model']])
            add_tensor(f'conv_blocks.{i}.norm2.weight', blk['norm2_weight'], [config['d_model']])
        
        # Attention blocks (GQA)
        attn_blocks = weights.get('attn_blocks', [])
        for i, blk in enumerate(attn_blocks):
            add_tensor(f'attn_blocks.{i}.norm1.weight', blk['norm1_weight'], [config['d_model']])
            add_tensor(f'attn_blocks.{i}.q.weight', blk['q_weight'], [config['d_model'], config['d_model']])
            n_kv = config.get('n_heads', 6) // 4 or 2
            d_head = config['d_model'] // config.get('n_heads', 6)
            add_tensor(f'attn_blocks.{i}.kv.weight', blk['kv_weight'],
                       [2 * n_kv * d_head, config['d_model']])
            add_tensor(f'attn_blocks.{i}.out.weight', blk['out_weight'],
                       [config['d_model'], config['d_model']])
            add_tensor(f'attn_blocks.{i}.norm2.weight', blk['norm2_weight'], [config['d_model']])
    else:
        # Original architecture (legacy)
        print("Detected: legacy architecture (single FFN, all-attention)")
        if 'norm1_weight' in weights:
            add_tensor('block.norm1.weight', weights['norm1_weight'], [config['d_model']])
        if 'attn_qkv_weight' in weights:
            add_tensor('block.attn.qkv.weight', weights['attn_qkv_weight'],
                       [3*config['d_model'], config['d_model']])
        if 'attn_out_weight' in weights:
            add_tensor('block.attn.out.weight', weights['attn_out_weight'],
                       [config['d_model'], config['d_model']])
        if 'norm2_weight' in weights:
            add_tensor('block.norm2.weight', weights['norm2_weight'], [config['d_model']])
        if 'ffn_gate_weight' in weights:
            add_tensor('block.ffn.gate.weight', weights['ffn_gate_weight'],
                       [config['d_ff'], config['d_model']])
        if 'ffn_val_weight' in weights:
            add_tensor('block.ffn.val.weight', weights['ffn_val_weight'],
                       [config['d_ff'], config['d_model']])
        if 'ffn_down_weight' in weights:
            add_tensor('block.ffn.down.weight', weights['ffn_down_weight'],
                       [config['d_model'], config['d_ff']])
    
    # Write GGUF file
    print(f"\nWriting GGUF: {gguf_path}")
    
    # Count metadata first
    extra_config_keys = [k for k in config.keys() 
                         if k not in ['vocab_size', 'd_model', 'n_heads', 'd_ff', 'n_layers', 
                                      'max_seq', 'n_conv_blocks', 'n_attn_blocks', 'n_experts',
                                      'conv_kernel', 'epochs', 'batch_size', 'lr']]
    n_metadata = 17 + len(extra_config_keys)  # 17 fixed keys + extras
    
    with open(gguf_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors)))  # tensor count
        f.write(struct.pack('<Q', n_metadata))  # metadata count (correct!)
        
        # Metadata: config keys
        write_metadata_kv(f, 'architecture', model.get('architecture', 'dquilane'), GGUF_TYPE_STRING)
        write_metadata_kv(f, 'techniques', json.dumps(model.get('techniques', [])), GGUF_TYPE_STRING)
        write_metadata_kv(f, 'vocab_size', config['vocab_size'], GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'd_model', config['d_model'], GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'n_heads', config.get('n_heads', 6), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'd_ff', config.get('d_ff', 384), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'n_layers', config.get('n_layers', 12), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'max_seq', config.get('max_seq', 80), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'n_conv_blocks', config.get('n_conv_blocks', 0), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'n_attn_blocks', config.get('n_attn_blocks', 0), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'n_experts', config.get('n_experts', 0), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'conv_kernel', config.get('conv_kernel', 4), GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'eos_id', eos_id, GGUF_TYPE_UINT32)
        write_metadata_kv(f, 'pad_id', stoi.get('<pad>', 0), GGUF_TYPE_UINT32)
        
        # Tokenizer (string-encoded)
        write_metadata_kv(f, 'tokenizer_stoi', json.dumps(stoi, ensure_ascii=False), GGUF_TYPE_STRING)
        write_metadata_kv(f, 'tokenizer_itos', json.dumps(itos, ensure_ascii=False), GGUF_TYPE_STRING)
        
        # AION engram
        write_metadata_kv(f, 'aion_engram', json.dumps(aion_engram, ensure_ascii=False), GGUF_TYPE_STRING)
        
        # Extra config keys
        for k, v in config.items():
            if k not in ['vocab_size', 'd_model', 'n_heads', 'd_ff', 'n_layers', 
                         'max_seq', 'n_conv_blocks', 'n_attn_blocks', 'n_experts',
                         'conv_kernel', 'epochs', 'batch_size', 'lr']:
                write_metadata_kv(f, f'config.{k}', str(v), GGUF_TYPE_STRING)
        
        # Compute tensor data offsets
        # First write tensor info
        tensor_info_offset = f.tell()
        # We need to know the data section start first
        # Compute total tensor info size
        # Each tensor info: name (string) + n_dims (uint32) + dims (uint64[]) + type (uint32) + offset (uint64)
        # + padding for alignment
        
        # Estimate info size
        info_size = 0
        for name, dims, _ in tensors:
            info_size += 8 + len(name.encode('utf-8'))  # name
            info_size += 4  # n_dims
            info_size += 8 * len(dims)  # dims
            info_size += 4  # type
            info_size += 8  # offset
        
        # Align data section to GGUF_ALIGNMENT
        data_start = tensor_info_offset + info_size
        padding = (GGUF_ALIGNMENT - (data_start % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        data_start += padding
        
        # Write tensor info
        current_offset = 0
        for name, dims, flat_data in tensors:
            write_string(f, name)
            f.write(struct.pack('<I', len(dims)))
            for d in dims:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', GGUF_TYPE_F32))  # F32 type
            f.write(struct.pack('<Q', current_offset))
            # Compute next offset (aligned)
            tensor_size = len(flat_data) * 4  # 4 bytes per F32
            current_offset += tensor_size
            # Align next tensor
            current_offset += (GGUF_ALIGNMENT - (current_offset % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
        
        # Pad to data_start
        current_pos = f.tell()
        if current_pos < data_start:
            f.write(b'\x00' * (data_start - current_pos))
        
        # Write tensor data
        for name, dims, flat_data in tensors:
            # Align to GGUF_ALIGNMENT
            current_pos = f.tell()
            expected_offset = data_start + (current_pos - data_start)
            # Actually current_pos should match the offset recorded
            # Pad if needed
            padding_needed = (GGUF_ALIGNMENT - (current_pos % GGUF_ALIGNMENT)) % GGUF_ALIGNMENT
            if padding_needed > 0:
                f.write(b'\x00' * padding_needed)
            
            # Write F32 data
            import array
            arr = array.array('f', flat_data)
            arr.tofile(f)
    
    file_size = os.path.getsize(gguf_path)
    print(f"\n✓ GGUF exported: {gguf_path}")
    print(f"  File size: {file_size / 1024:.1f} KB ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Format: GGUF v{GGUF_VERSION}")
    print(f"  Precision: F32 (no quantization)")
    print(f"  Architecture: {model.get('architecture', 'unknown')}")
    print(f"  Techniques: {', '.join(model.get('techniques', []))}")
    print(f"  AION Engram: {len(aion_engram)} Q&A pairs")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python gguf_export.py <model.json> <output.gguf>")
        sys.exit(1)
    export_gguf(sys.argv[1], sys.argv[2])
