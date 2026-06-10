from typing import List, Optional
from openai import OpenAI


class LLMClient:
    """LLM API客户端"""

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化LLM客户端

        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
        """
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置，请在 .env 中配置")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大token数

        Returns:
            模型回复文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def chat_with_context(
        self,
        question: str,
        context: str,
        system_prompt: str = "你是一个有帮助的助手。",
    ) -> str:
        """
        带上下文的聊天

        Args:
            question: 用户问题
            context: 上下文内容
            system_prompt: 系统提示

        Returns:
            模型回复
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"参考信息：\n{context}\n\n问题：{question}"},
        ]
        return self.chat(messages)
