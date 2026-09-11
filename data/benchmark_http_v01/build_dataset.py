"""Build the deterministic unseen-family HTTP/API transfer benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


PATH = {"type": "string", "pattern": "^/"}
STATUS_ARRAY = {
    "type": "array",
    "items": {"type": "integer", "minimum": 100, "maximum": 599},
    "minItems": 1,
}
ON_UNACCEPTED = {"type": "string", "enum": ["raise", "return_body"]}
JSON_OBJECT = {"type": "object"}


SKILLS = [
    {
        "id": "shttp01",
        "name": "http_get_json",
        "brief_description": "从指定 API 路径读取 JSON 资源。",
        "detailed_description": (
            "发起一次 GET。默认仅接受 200/201/202/204；若任务限定成功状态，"
            "传 accepted_statuses。若未接受状态必须原样返回且不自动重试，"
            "传 on_unaccepted=return_body；否则抛出错误。"
        ),
        "category": "http_read",
        "parameters": object_schema(
            {
                "path": PATH,
                "accepted_statuses": STATUS_ARRAY,
                "on_unaccepted": ON_UNACCEPTED,
            },
            ["path"],
        ),
        "returns": {"type": "string", "description": "JSON 或原始响应正文"},
        "tags": ["HTTP", "GET", "JSON", "API"],
        "examples": [
            '严格状态失败时：{"path":"/limits","accepted_statuses":[200],"on_unaccepted":"return_body"}'
        ],
        "dependencies": [],
        "metadata": {"permission": "http:read"},
    },
    {
        "id": "shttp02",
        "name": "http_get_text",
        "brief_description": "从指定 API 路径读取纯文本正文。",
        "detailed_description": "发起一次 GET 并返回文本；适用于非 JSON 内容。",
        "category": "http_read",
        "parameters": object_schema(
            {"path": PATH, "accepted_statuses": STATUS_ARRAY}, ["path"]
        ),
        "returns": {"type": "string"},
        "tags": ["HTTP", "GET", "text"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:read"},
    },
    {
        "id": "shttp03",
        "name": "http_head_status",
        "brief_description": "用 HEAD 检查 API 路径并返回状态码。",
        "detailed_description": "只发起一次 HEAD，不下载响应正文，返回整数状态码。",
        "category": "http_read",
        "parameters": object_schema({"path": PATH}, ["path"]),
        "returns": {"type": "integer", "minimum": 100, "maximum": 599},
        "tags": ["HTTP", "HEAD", "status", "health"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:read"},
    },
    {
        "id": "shttp04",
        "name": "http_issue_token",
        "brief_description": "为给定 scope 从本地认证端点签发访问令牌。",
        "detailed_description": "POST /oauth/token，并从 JSON 响应中只返回 access_token 字符串。",
        "category": "http_auth",
        "parameters": object_schema({"scope": {"type": "string"}}, ["scope"]),
        "returns": {"type": "string", "description": "Bearer token"},
        "tags": ["HTTP", "OAuth", "token", "authentication"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:auth"},
    },
    {
        "id": "shttp05",
        "name": "http_get_bearer_json",
        "brief_description": "携带 Bearer token 读取受保护 JSON API。",
        "detailed_description": "使用 Authorization: Bearer <token> 发起一次 GET 并返回 JSON。",
        "category": "http_auth",
        "parameters": object_schema(
            {"path": PATH, "token": {"type": "string"}}, ["path", "token"]
        ),
        "returns": {"type": "string", "description": "JSON 响应"},
        "tags": ["HTTP", "GET", "Bearer", "protected"],
        "examples": [
            'token 可使用上一步输出：{"path":"/reports/daily","token":"$last_output"}'
        ],
        "dependencies": ["shttp04"],
        "metadata": {"permission": "http:auth", "depends_on": ["shttp04"]},
    },
    {
        "id": "shttp06",
        "name": "http_post_json",
        "brief_description": "带幂等键向 API 路径提交 JSON。",
        "detailed_description": (
            "只发起一次 POST。默认接受 200/201/202/204，成功时返回包含 status 与 data 的包装 JSON。"
            "若任务说仅当特定状态成功，应传 accepted_statuses；若未接受状态要求原样返回且不自动重试，"
            "同时传 on_unaccepted=return_body，此时直接返回服务正文，不加包装。"
        ),
        "category": "http_write",
        "parameters": object_schema(
            {
                "path": PATH,
                "payload": JSON_OBJECT,
                "idempotency_key": {"type": "string", "minLength": 1},
                "accepted_statuses": STATUS_ARRAY,
                "on_unaccepted": ON_UNACCEPTED,
            },
            ["path", "payload", "idempotency_key"],
        ),
        "returns": {"type": "string"},
        "tags": ["HTTP", "POST", "JSON", "idempotency"],
        "examples": [
            '仅接受 201：{"accepted_statuses":[201],"on_unaccepted":"return_body"}'
        ],
        "dependencies": [],
        "metadata": {"permission": "http:write"},
    },
    {
        "id": "shttp07",
        "name": "http_put_json",
        "brief_description": "通过 PUT 将完整 JSON 表示写入指定 API 路径。",
        "detailed_description": "发起一次 PUT；用于完整替换资源，不用于局部字段更新。",
        "category": "http_write",
        "parameters": object_schema(
            {"path": PATH, "payload": JSON_OBJECT}, ["path", "payload"]
        ),
        "returns": {"type": "string", "description": "JSON 响应"},
        "tags": ["HTTP", "PUT", "replace", "JSON"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:write"},
    },
    {
        "id": "shttp08",
        "name": "http_patch_json",
        "brief_description": "通过 PATCH 对 API 资源做 JSON 局部更新。",
        "detailed_description": "发起一次 PATCH；只发送待变更字段，不替换未提供字段。",
        "category": "http_write",
        "parameters": object_schema(
            {"path": PATH, "patch": JSON_OBJECT}, ["path", "patch"]
        ),
        "returns": {"type": "string", "description": "JSON 响应"},
        "tags": ["HTTP", "PATCH", "partial update", "JSON"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:write"},
    },
    {
        "id": "shttp09",
        "name": "http_delete_resource",
        "brief_description": "通过 DELETE 删除指定 API 资源并返回状态码。",
        "detailed_description": "发起一次 DELETE；只接受 200/202/204，返回服务状态码。",
        "category": "http_write",
        "parameters": object_schema({"path": PATH}, ["path"]),
        "returns": {"type": "integer"},
        "tags": ["HTTP", "DELETE", "resource"],
        "examples": [],
        "dependencies": [],
        "metadata": {"permission": "http:write"},
    },
]


def request(method: str, path: str, status: int, *, body: Any = None, headers: dict | None = None) -> dict:
    return {
        "method": method,
        "path": path,
        "headers": headers or {},
        "json": body,
        "response_status": status,
    }


def http_state(requests: list[dict], routes: list[dict]) -> dict:
    return {
        "requests": requests,
        "route_call_counts": {
            f"{route['method']} {route['path']}": sum(
                1 for item in requests
                if item["method"] == route["method"] and item["path"] == route["path"]
            )
            for route in routes
        },
    }


def make_task(
    task_id: str,
    split: str,
    template_id: str,
    instruction: str,
    expected_skills: list[str],
    inputs: dict[str, Any],
    expected_output: Any,
    routes: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[str, dict[str, Any]]]:
    fixture_id = f"fixture-{task_id}"
    task = {
        "id": task_id,
        "instruction": instruction,
        "expected_skills": expected_skills,
        "expected_skill_sequence": expected_skills,
        "inputs": inputs,
        "metadata": {
            "split": split,
            "family": "local_http_api",
            "template_id": template_id,
            "slice": {"context_budget_tokens": 2600},
            "environment_fixture": {"http_fixture_id": fixture_id},
        },
        "evaluation": {
            "verifier_type": "http_state",
            "expected_output": expected_output,
            "expected_state": http_state(requests, routes),
        },
    }
    return task, (fixture_id, {"type": "local_http_api", "routes": routes})


TASK_SPECS = [
    make_task(
        "thttpd01", "dev", "dev-catalog-lookup",
        "读取任务输入 path 指向的商品 JSON，返回完整对象。",
        ["shttp01"], {"path": "/catalog/SKU-7"},
        {"name": "蓝牙键盘", "sku": "SKU-7", "stock": 12},
        [{"method": "GET", "path": "/catalog/SKU-7", "status": 200,
          "response_json": {"sku": "SKU-7", "name": "蓝牙键盘", "stock": 12}}],
        [request("GET", "/catalog/SKU-7", 200)],
    ),
    make_task(
        "thttpd02", "dev", "dev-service-probe",
        "只用 HEAD 检查任务输入 path 的服务状态，并返回状态码。",
        ["shttp03"], {"path": "/health/worker"}, 204,
        [{"method": "HEAD", "path": "/health/worker", "status": 204,
          "response_json": {"ok": True}}],
        [request("HEAD", "/health/worker", 204)],
    ),
    make_task(
        "thttpd03", "dev", "dev-protected-report",
        "先按任务输入 scope 取得访问令牌，再把上一步令牌用于读取 path 的受保护日报 JSON。",
        ["shttp04", "shttp05"],
        {"scope": "reports:read", "path": "/reports/daily"},
        {"date": "2026-09-03", "total": 41},
        [
            {"method": "POST", "path": "/oauth/token", "status": 200,
             "expected_json": {"scope": "reports:read"},
             "response_json": {"access_token": "tok-dev-report"}},
            {"method": "GET", "path": "/reports/daily", "status": 200,
             "required_headers": {"authorization": "Bearer tok-dev-report"},
             "response_json": {"date": "2026-09-03", "total": 41}},
        ],
        [
            request("POST", "/oauth/token", 200, body={"scope": "reports:read"}),
            request("GET", "/reports/daily", 200,
                    headers={"authorization": "Bearer tok-dev-report"}),
        ],
    ),
    make_task(
        "thttpd04", "dev", "dev-strict-create-status",
        "提交任务输入 payload。仅当状态码为 201 才算成功；其他状态必须原样返回响应正文，并且不自动重试。",
        ["shttp06"],
        {"path": "/jobs", "payload": {"kind": "export"}, "idempotency_key": "job-dev-41"},
        '{"error": "queued", "request_id": "q-dev"}',
        [{"method": "POST", "path": "/jobs", "status": 202,
          "required_headers": {"idempotency-key": "job-dev-41"},
          "expected_json": {"kind": "export"},
          "response_json": {"error": "queued", "request_id": "q-dev"}}],
        [request("POST", "/jobs", 202, body={"kind": "export"},
                 headers={"idempotency-key": "job-dev-41"})],
    ),
    make_task(
        "thttpd05", "dev", "dev-profile-replace",
        "用任务输入 payload 完整替换 path 的资料，并返回服务 JSON。",
        ["shttp07"],
        {"path": "/profiles/u7", "payload": {"name": "林青", "level": 3}},
        {"id": "u7", "name": "林青", "level": 3},
        [{"method": "PUT", "path": "/profiles/u7", "status": 200,
          "expected_json": {"name": "林青", "level": 3},
          "response_json": {"id": "u7", "name": "林青", "level": 3}}],
        [request("PUT", "/profiles/u7", 200, body={"name": "林青", "level": 3})],
    ),
    make_task(
        "thttpd06", "dev", "dev-rate-limit-body",
        "请求任务输入 path 一次。若状态不是 200，原样返回错误正文，不自动重试。",
        ["shttp01"], {"path": "/limits/current"},
        '{"error": "rate_limited", "retry_after": 30}',
        [{"method": "GET", "path": "/limits/current", "status": 429,
          "response_json": {"error": "rate_limited", "retry_after": 30}}],
        [request("GET", "/limits/current", 429)],
    ),
    make_task(
        "thttpc01", "confirmation", "confirmation-shipment-fetch",
        "获取任务输入 path 对应的运单 API JSON，并完整返回。",
        ["shttp01"], {"path": "/shipments/SHP-88"},
        {"carrier": "北辰", "id": "SHP-88", "state": "in_transit"},
        [{"method": "GET", "path": "/shipments/SHP-88", "status": 200,
          "response_json": {"id": "SHP-88", "carrier": "北辰", "state": "in_transit"}}],
        [request("GET", "/shipments/SHP-88", 200)],
    ),
    make_task(
        "thttpc02", "confirmation", "confirmation-readiness-probe",
        "对任务输入 path 做 HEAD 探测，只返回 HTTP 状态码。",
        ["shttp03"], {"path": "/ready/search"}, 200,
        [{"method": "HEAD", "path": "/ready/search", "status": 200,
          "response_json": {"ready": True}}],
        [request("HEAD", "/ready/search", 200)],
    ),
    make_task(
        "thttpc03", "confirmation", "confirmation-protected-invoice",
        "使用输入 scope 先签发令牌，然后将该输出作为 token 读取输入 path 的受保护账单 JSON。",
        ["shttp04", "shttp05"],
        {"scope": "invoices:read", "path": "/invoices/INV-9"},
        {"amount": 880, "id": "INV-9", "paid": False},
        [
            {"method": "POST", "path": "/oauth/token", "status": 200,
             "expected_json": {"scope": "invoices:read"},
             "response_json": {"access_token": "tok-confirm-invoice"}},
            {"method": "GET", "path": "/invoices/INV-9", "status": 200,
             "required_headers": {"authorization": "Bearer tok-confirm-invoice"},
             "response_json": {"id": "INV-9", "amount": 880, "paid": False}},
        ],
        [
            request("POST", "/oauth/token", 200, body={"scope": "invoices:read"}),
            request("GET", "/invoices/INV-9", 200,
                    headers={"authorization": "Bearer tok-confirm-invoice"}),
        ],
    ),
    make_task(
        "thttpc04", "confirmation", "confirmation-strict-import-status",
        "向 path 提交 payload。仅当返回 201 时认定提交成功；否则原样返回响应正文，绝不自动重试。",
        ["shttp06"],
        {"path": "/imports", "payload": {"source": "ledger"}, "idempotency_key": "import-confirm-9"},
        '{"error": "accepted_later", "request_id": "q-confirm"}',
        [{"method": "POST", "path": "/imports", "status": 202,
          "required_headers": {"idempotency-key": "import-confirm-9"},
          "expected_json": {"source": "ledger"},
          "response_json": {"error": "accepted_later", "request_id": "q-confirm"}}],
        [request("POST", "/imports", 202, body={"source": "ledger"},
                 headers={"idempotency-key": "import-confirm-9"})],
    ),
    make_task(
        "thttpc05", "confirmation", "confirmation-settings-patch",
        "把任务输入 patch 作为局部更新提交到 path，并返回 API JSON。",
        ["shttp08"],
        {"path": "/settings/team-a", "patch": {"timezone": "Asia/Shanghai"}},
        {"team": "team-a", "timezone": "Asia/Shanghai", "unchanged": 4},
        [{"method": "PATCH", "path": "/settings/team-a", "status": 200,
          "expected_json": {"timezone": "Asia/Shanghai"},
          "response_json": {"team": "team-a", "timezone": "Asia/Shanghai", "unchanged": 4}}],
        [request("PATCH", "/settings/team-a", 200,
                 body={"timezone": "Asia/Shanghai"})],
    ),
    make_task(
        "thttpc06", "confirmation", "confirmation-maintenance-body",
        "访问任务输入 path 恰好一次；如果状态不是 200，保持错误正文原样返回，不自动重试。",
        ["shttp01"], {"path": "/status/dependency"},
        '{"error": "maintenance", "window": "02:00-02:15"}',
        [{"method": "GET", "path": "/status/dependency", "status": 503,
          "response_json": {"error": "maintenance", "window": "02:00-02:15"}}],
        [request("GET", "/status/dependency", 503)],
    ),
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    tasks = [task for task, _ in TASK_SPECS]
    fixtures = dict(fixture for _, fixture in TASK_SPECS)
    write_jsonl(ROOT / "skills.jsonl", SKILLS)
    write_jsonl(ROOT / "tasks.jsonl", tasks)
    (ROOT / "environment_fixtures.json").write_text(
        json.dumps(fixtures, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(SKILLS)} skills, {len(tasks)} tasks, {len(fixtures)} fixtures")


if __name__ == "__main__":
    main()
