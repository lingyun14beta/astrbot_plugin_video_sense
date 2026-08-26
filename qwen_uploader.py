"""百炼（DashScope）大视频上传模块：本地文件 → 免费临时 URL（oss://）。

官方文档（https://help.aliyun.com/zh/model-studio/get-temporary-file-url）要点：
- 先 GET /api/v1/uploads?action=getPolicy&model=<模型名> 获取上传凭证；
- 再把文件 POST 到 upload_host（OSS 临时存储空间），得到 oss://<key> 前缀的临时 URL；
- 临时 URL 有效期 48 小时；文件与模型绑定（上传与调用必须同一模型）；
- 上传接口单文件不超过 1GB；凭证接口限流 100 QPS，官方提示勿用于生产环境。

本模块只负责上传，请求体构造与协议选择见 gemini_client.py。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp

_MB = 1024 * 1024
# 上传接口限制：单文件不超过 1GB
_MAX_UPLOAD_BYTES = 1024 * _MB

_HTTP_OK = 200

# base_url 可能带这些尾部，推导 /api/v1/uploads 时先剥离
_TAIL_SUFFIXES = (
    "/compatible-mode/v1",
    "/chat/completions",
    "/api/v1",
    "/v1beta/openai",
    "/v1beta",
    "/v1",
)


class QwenUploadError(Exception):
    """百炼临时 URL 上传失败，message 可直接透传给用户。"""


async def upload_video_to_temp_url(
    api_key: str,
    base_url: str,
    model: str,
    file_path: str,
    mime_type: str,
    filename: str = "",
    timeout: int = 300,
) -> str:
    """上传视频到百炼免费临时存储空间，返回 oss:// 临时 URL。

    Args:
        api_key: 阿里云百炼 API Key。
        base_url: OpenAI 兼容接口地址（如
            https://dashscope.aliyuncs.com/compatible-mode/v1）。
        model: 调用模型名称（如 qwen-vl-max）。文件与模型绑定，必须与后续
            调用使用的模型一致。
        file_path: 本地视频文件路径。
        mime_type: 视频 MIME 类型（如 video/mp4）。
        filename: 上传文件名，留空使用文件路径中的名称。
        timeout: 上传请求总超时（秒）。

    Returns:
        oss://<key> 形式的临时 URL，可直接作为 video_url 的 url 使用。

    Raises:
        QwenUploadError: Key/模型缺失、文件不存在、超出大小限制或上传失败。
    """
    if not api_key:
        raise QwenUploadError("未配置 API Key，请在插件设置中填写。")
    if not model:
        raise QwenUploadError(
            "未指定模型（model），临时 URL 上传要求文件与调用模型一致。",
        )
    p = Path(file_path)
    if not p.is_file():
        raise QwenUploadError(f"文件不存在或无法访问：{file_path}")
    size = p.stat().st_size
    if size <= 0:
        raise QwenUploadError("视频文件为空，无法上传。")
    if size > _MAX_UPLOAD_BYTES:
        raise QwenUploadError(
            f"文件 {size / _MB:.1f} MB 超过百炼临时存储单文件上限 1GB，"
            "无法通过临时 URL 上传，请压缩或截取视频片段后重试。",
        )

    policy_url = build_policy_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(
            timeout=client_timeout, trust_env=False
        ) as session:
            async with session.get(
                policy_url,
                headers=headers,
                params={"action": "getPolicy", "model": model},
            ) as resp:
                raw = await resp.text()
                if resp.status != _HTTP_OK:
                    msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                    raise QwenUploadError(
                        f"获取上传凭证失败（{resp.status}）：{msg}",
                    )
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise QwenUploadError(
                        f"上传凭证响应解析失败：{e}",
                    ) from e
                if not isinstance(parsed, dict) or not isinstance(
                    parsed.get("data"), dict
                ):
                    raise QwenUploadError("上传凭证响应格式异常（缺少 data）。")
                policy = parsed["data"]
                if policy.get("error"):
                    raise QwenUploadError(f"获取上传凭证失败：{policy['error']}")
                upload_host = (policy.get("upload_host") or "").strip()
                if not upload_host:
                    raise QwenUploadError("上传凭证缺少 upload_host。")
                for field in ("policy", "signature", "oss_access_key_id"):
                    if not policy.get(field):
                        raise QwenUploadError(
                            f"上传凭证缺少 {field}，无法上传。",
                        )

            upload_dir = (policy.get("upload_dir") or "").strip("/")
            key = f"{upload_dir}/{filename or p.name}".lstrip("/")
            # 用文件句柄流式上传，避免整文件读入内存（上限 1GB）
            with p.open("rb") as fh:
                form = aiohttp.FormData()
                form.add_field("OSSAccessKeyId", policy["oss_access_key_id"])
                form.add_field("signature", policy["signature"])
                form.add_field("policy", policy["policy"])
                if policy.get("x_oss_object_acl"):
                    form.add_field("x-oss-object-acl", policy["x_oss_object_acl"])
                if policy.get("x_oss_forbid_overwrite"):
                    form.add_field(
                        "x-oss-forbid-overwrite", policy["x_oss_forbid_overwrite"]
                    )
                form.add_field("key", key)
                form.add_field("success_action_status", "200")
                form.add_field(
                    "file",
                    fh,
                    filename=filename or p.name,
                    content_type=mime_type,
                )
                # OSS 表单上传无需 Bearer 鉴权（凭证在表单字段中）
                async with session.post(upload_host, data=form) as resp:
                    raw = await resp.text()
                    if resp.status != _HTTP_OK:
                        msg = _extract_error_message(raw) or f"HTTP {resp.status}"
                        raise QwenUploadError(f"上传文件失败（{resp.status}）：{msg}")
    except aiohttp.ClientError as e:
        raise QwenUploadError(f"网络请求异常：{e}") from e

    return f"oss://{key}"


def build_policy_url(base_url: str) -> str:
    """由 OpenAI 兼容 base_url 推导 /api/v1/uploads 凭证接口地址。

    Args:
        base_url: 配置的 Base URL（如
            https://dashscope.aliyuncs.com/compatible-mode/v1）。

    Returns:
        凭证接口地址，如 https://dashscope.aliyuncs.com/api/v1/uploads。
    """
    base = (base_url or "").strip().rstrip("/")
    for suffix in _TAIL_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.endswith("/api/v1/uploads"):
        return base
    return f"{base}/api/v1/uploads"


def _extract_error_message(raw: str) -> str:
    """从响应文本提取用户可读的错误信息。"""
    if not raw:
        return ""
    if raw.lstrip().startswith("<"):
        return "网关返回 HTML 错误页（可能为反向代理限制请求大小，或上游服务错误）。"
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            err = data.get("error") or {}
            if isinstance(err, dict):
                return err.get("message", "")
            return str(err)
    except Exception:
        pass
    return raw[:200]
