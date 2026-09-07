# Third-party notices

## vLLM GGUF plugin

The dense and tiered kernels build against GGUF quant-format declarations and
dot-product primitives from the vLLM GGUF plugin:

https://github.com/vllm-project/vllm-gguf-plugin

The plugin is licensed under Apache License 2.0. R9V's published integration
fork retains that license and identifies its modifications.

## llama.cpp / ggml

The plugin's GGUF implementation includes code copied or adapted from
llama.cpp/ggml, historically identified in the source as revision `b2899`.
R9V therefore retains the upstream MIT attribution for the quant structures,
dequantization helpers, and vector-dot implementations used by these kernels.

Project: https://github.com/ggml-org/llama.cpp

MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
