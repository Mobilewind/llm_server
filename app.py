# app.py
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import uvicorn
import os
import torch

# 使用正确的模型名称
model_name = "warshanks/Qwen3-4B-Instruct-2507-AWQ"

# 配置异步引擎参数
engine_args = AsyncEngineArgs(
    model=model_name,
    trust_remote_code=True,  # Qwen模型需要此选项
    tensor_parallel_size=1,  # 如果使用多GPU，可以调整此参数
    #gpu_memory_utilization=0.9,  # GPU内存使用率
    gpu_memory_utilization=0.7,  # GPU内存使用率
    #max_model_len=16384,  # 根据你的需要调整最大模型长度
    max_model_len=8192,  # 减少最大序列长度
    quantization="compressed-tensors",           # 使用模型配置的量化方法
    dtype="half",                # 使用半精度浮点数
    disable_custom_all_reduce=True,
    enforce_eager=True,          # 禁用图优化，减少内存占用    
)

# 创建异步引擎
try:
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("vLLM engine initialized successfully!")
except Exception as e:
    print(f"vLLM initialization failed: {e}")
    # 如果 vLLM 失败，回退到标准 transformers
    print("Falling back to standard transformers...")
    engine = None

app = FastAPI(title="LLM API Server", version="1.0")

class GenerateRequest(BaseModel):
    input: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1

class GenerateResponse(BaseModel):
    generated_text: str
    model: str
    device: str = "cuda"

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        # 为 Qwen 模型构建对话格式
        messages = [
            {"role": "user", "content": request.input}
        ]
        
        # 使用vLLM引擎生成
        sampling_params = SamplingParams(
            n=1,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_new_tokens,
            repetition_penalty=request.repetition_penalty,
        )
        
        # 构建提示词，vLLM会自动处理聊天模板
        # 对于Qwen模型，我们可以直接使用消息格式
        prompt = messages[0]["content"]
        
        # 开始生成
        request_id = random_uuid()
        results_generator = engine.generate(
            prompt, sampling_params, request_id
        )
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output:
            generated_text = final_output.outputs[0].text
            return GenerateResponse(
                generated_text=generated_text.strip(),
                model=model_name,
                device="cuda"
            )
        else:
            raise HTTPException(status_code=500, detail="No output generated")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
        
# 健康检查端点...    
@app.get("/health")
async def health_check():
    gpu_status = "available" if torch.cuda.is_available() else "unavailable"
    engine_status = "initialized" if engine is not None else "failed"
    
    return {
        "status": "healthy",
        "gpu": gpu_status,
        "engine": engine_status,
        "model": model_name
    }
  

# 健康检查端点...
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)