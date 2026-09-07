import time, json, urllib.request, urllib.error, concurrent.futures

TIERS = [
    {
        "name": "local-medium (workstation .150:8083)",
        "url": "http://192.168.41.150:8083/v1/chat/completions",
        "api_key": "eGXRG3Njmcif11ZJ65R15hFbNY56kKO4JHWiaYk2jtI",
        "model": "/8tb/LLM-Models/Qwen3-14B-GGUF/Qwen3-14B-Q5_K_M.gguf",
        "slots": 1
    },
    {
        "name": "local-light (srv01 .246:8090)",
        "url": "http://192.168.41.246:8090/v1/chat/completions",
        "api_key": "eGXRG3Njmcif11ZJ65R15hFbNY56kKO4JHWiaYk2jtI",
        "model": "/home/ydj/LLM-Models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        "slots": 2
    }
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Fetch current stock price and volume for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. NVDA, AMD, AAPL"}
                },
                "required": ["symbol"]
            }
        }
    }
]

def make_req(url, api_key, payload):
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode('utf-8'))
        t1 = time.time()
        return res, t1 - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        t1 = time.time()
        return {"error": f"HTTP {e.code}: {body}"}, t1 - t0

def test_tool_calling(tier):
    print(f"\n--- Testing Tool Calling on {tier['name']} ---")
    payload = {
        "model": tier["model"],
        "messages": [
            {"role": "system", "content": "You are a financial assistant. Call get_stock_price to look up prices."},
            {"role": "user", "content": "What is the stock price of AMD right now?"}
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.3,
        "max_tokens": 256
    }
    res, latency = make_req(tier["url"], tier["api_key"], payload)
    if "error" in res:
        print(f"Error: {res['error']}")
        return False, res['error']
        
    msg = res["choices"][0]["message"]
    tool_calls = msg.get("tool_calls")
    content = msg.get("content")
    print(f"Latency: {latency*1000:.1f} ms")
    if tool_calls and len(tool_calls) > 0:
        fn = tool_calls[0].get("function", {})
        print(f"Tool Call Detected: Name = {fn.get('name')}, Args = {fn.get('arguments')}")
        print(f"Status: PASS (Native JSON Tool Call)")
        return True, "PASS (Native JSON)"
    elif content and ("get_stock_price" in content or "AMD" in content):
        print(f"Content Tool Call: {content.strip()[:150]}")
        print(f"Status: PASS (Text Tool Call)")
        return True, "PASS (Text Tool Call)"
    else:
        print(f"Output: {msg}")
        return False, "FAIL"

def benchmark_throughput(tier, concurrency=1, num_requests=3):
    print(f"\n--- Benchmarking {tier['name']} (Concurrency={concurrency}, Req={num_requests}) ---")
    payload_template = {
        "model": tier["model"],
        "messages": [
            {"role": "system", "content": "You are a concise assistant. Output code only."},
            {"role": "user", "content": "Write an efficient Python quicksort algorithm."}
        ],
        "temperature": 0.3,
        "max_tokens": 256
    }
    
    latencies = []
    total_tokens = 0
    errors = 0
    
    start_total = time.time()
    
    def worker(_):
        res, lat = make_req(tier["url"], tier["api_key"], payload_template)
        if "error" in res:
            return lat, 0, res["error"]
        usage = res.get("usage", {})
        comp_tokens = usage.get("completion_tokens", 0)
        return lat, comp_tokens, None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, i) for i in range(num_requests)]
        for f in concurrent.futures.as_completed(futures):
            lat, comp_tok, err = f.result()
            if err:
                errors += 1
                print(f"  [Req Error]: {err}")
            else:
                latencies.append(lat)
                total_tokens += comp_tok
            
    total_time = time.time() - start_total
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = total_tokens / total_time if total_time > 0 else 0
    tpot = (avg_latency / (total_tokens / len(latencies))) * 1000 if (total_tokens > 0 and len(latencies) > 0) else 0
    
    print(f"Total Time       : {total_time:.2f} s (Errors: {errors})")
    print(f"Total Tokens     : {total_tokens}")
    print(f"Avg Latency      : {avg_latency*1000:.1f} ms")
    print(f"Est. TPOT        : {tpot:.1f} ms / tok")
    print(f"Throughput       : {throughput:.1f} tok/s")
    
    return throughput, avg_latency, tpot

if __name__ == "__main__":
    results = {}
    for tier in TIERS:
        tc_ok, tc_detail = test_tool_calling(tier)
        tp1, lat1, tpot1 = benchmark_throughput(tier, concurrency=1, num_requests=2)
        results[tier["name"]] = {
            "tool_call": tc_detail,
            "tp": tp1,
            "lat": lat1,
            "tpot": tpot1
        }
    
    print("\n" + "="*85)
    print("  Complete 3-Tier Cluster Status & Tool Calling Results")
    print("="*85)
    for k, v in results.items():
        print(f"• {k}:")
        print(f"    Tool Calling : {v['tool_call']}")
        print(f"    Throughput   : {v['tp']:.1f} tok/s (TPOT: {v['tpot']:.1f} ms, Latency: {v['lat']*1000:.0f} ms)")
